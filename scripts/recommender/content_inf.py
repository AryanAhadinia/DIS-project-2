import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import wandb
from src.matrix_factorization.pq import PQ
from utils import get_train_test

from src.bm25.bm25 import BM25

if __name__ == "__main__":
    """
    Script for training and testing the hybrid model using both matrix factorization and content-based filtering (BM25).
    Command-line Arguments:
        --token_dir (Path): Path to the directory containing the tokenized data (default is 'data/tokenized_data').
        --output_dir (Path): Path to the directory where the model will be saved (default is 'data').
        --data_dir (Path): Path to the directory containing the data (default is 'data').
        --d (int): Dimension of the latent space (default is 8).
        --lr (float): Learning rate (default is 0.004).
        --P_lambda (float): Regularization parameter for P (default is 0.494).
        --Q_lambda (float): Regularization parameter for Q (default is 0.132).
        --n_iter (int): Number of iterations (default is 20).
        --wandb_key (str): API key for wandb (default is '').
        --test (bool): Whether to test the model (default is False).
        --wandb_group (str): Group name for wandb (default is 'PQ').
        --submit (bool): Whether to create a submission (default is False).
        --k (int): Number of similar items to consider in the content-based filtering (default is 20).
        --skip_train (bool): Whether to skip training and load the model from the output directory (default is False).
        --content_user (bool): Whether to use content-based filtering for users (default is False).
        --k_ (int): Number of similar users to consider in the user content-based filtering (default is 300).
        --seed (int): Seed for random number generator (default is 0).
        --include_authors (bool): Whether to include authors in the content-based filtering (default is False).
        Output:
        Saves a pickled matrix factorization model to the output directory as 'pq_model.pkl'.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--token_dir", type=Path, default = "data/tokenized_data")
    parser.add_argument("--output_dir", type=Path, default = "data")
    parser.add_argument("--data_dir", type=Path, default="data")
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--P_lambda", type=float, default=0.494)
    parser.add_argument("--Q_lambda", type=float, default=0.132)
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--wandb_key", type=str, default="")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--wandb_group", type=str, default="PQ")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--content_user", action="store_true")
    parser.add_argument("--k_", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_authors", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.wandb_key != "":
        wandb.login(key=args.wandb_key)
    else:
        wandb.login()
    wandb.init(project="DIS2", group=args.wandb_group, config=vars(args))

    with open(f'{args.token_dir}/tokens.pkl', "rb") as f:
        docs = pickle.load(f)

    with open(f'{args.token_dir}/book_ids.pkl', "rb") as f:
        book_ids = np.array(pickle.load(f))

    with open('data/tokenized_data/authors.pkl', "rb") as f:
        list_authors = np.array(pickle.load(f))

    lang_params = {
        "k1": 1.4, "b": 0.5
    }

    # Create and fit the BM25 model
    bm25_ind = BM25(k1=lang_params["k1"], b=lang_params["b"])
    bm25_ind.fit(docs)

    data = pd.read_csv(args.data_dir / "train.csv")

    train_data, test_data = get_train_test(data, args.test)

    if args.skip_train:
        with open(args.output_dir / "pq_model.pkl", "rb") as f:
            pq_model = pickle.load(f)
    else:
        # Create matrix factorization model and fit it
        pq_model = PQ(len(data["user_id"].unique()), len(data["book_id"].unique()),
                      d=args.d, lr=args.lr, P_lambda=args.P_lambda, Q_lambda=args.Q_lambda)
        print("Fitting model")
        loss_train, loss_test = pq_model.fit(train_data, args.n_iter, test_data=test_data, log_wandb=True)
        if args.output_dir is not None:
            with open(args.output_dir / "pq_model.pkl", "wb") as f:
                pickle.dump(pq_model, f)

    R = pq_model.P @ pq_model.Q.T
    R[~np.isnan(train_data)] = train_data[~np.isnan(train_data)]
    if args.test:
        print('base: ', np.sqrt(np.nanmean((test_data - R) ** 2)))

    user_id = data['user_id'].unique()
    item_id = data['book_id'].unique()

    user_map = {u: i for i, u in enumerate(user_id)}
    item_map = {b: j for j, b in enumerate(item_id)}

    book_id_to_index = {book_id: i for i, book_id in enumerate(book_ids)}

    Q_ = pq_model.Q.copy()
    for b_id in tqdm(item_id, total=len(item_id)):
        if b_id not in book_id_to_index:
            continue
        idx = book_id_to_index[b_id]
        inds_, scores_ = bm25_ind.match(docs[idx], k=args.k)
        if args.include_authors:
            author = list_authors[idx]
            ind_same_authors = np.where(list_authors == author)[0]
            inds_same_authors = np.array([int(id) for id in ind_same_authors if id not in inds_])
            if len(inds_same_authors) > 0:
                inds_ = np.concatenate((inds_, inds_same_authors))
                scores_ = np.concatenate((scores_, np.ones(len(ind_same_authors)) * max(scores_) / 2))
        inds = []
        scores = []
        for i, s in zip(inds_, scores_):
            if book_ids[i] in item_map:
                inds.append(item_map[book_ids[i]])
                scores.append(s)
        if len(inds) == 0:
            continue

        scores = np.array(scores) / np.sum(scores)
        Q_[item_map[b_id]] = np.sum(pq_model.Q[inds] * scores[:, np.newaxis], axis=0)

    print("Filling matrix R_")
    R_ = pq_model.P @ Q_.T
    R_[~np.isnan(train_data)] = train_data[~np.isnan(train_data)]
    if args.content_user:
        R_norm = R_ / np.linalg.norm(R_, axis=1)[:, np.newaxis]
        all_scores = R_norm @ R_norm.T

    if args.test:
        print('testing')
        print(np.sqrt(np.nanmean((test_data - R_) ** 2)))

        if args.content_user:
            k_ = args.k_
            R_f = 0.9 * R_
            for u_ in tqdm(range(len(R_f))):
                most_similar_users = np.argsort(all_scores[u_])[-k_-1:-1]
                other_rating = np.sum(R_[most_similar_users, :] * all_scores[u_, most_similar_users][:, np.newaxis], axis=0) / np.sum(all_scores[u_, most_similar_users])
                R_f[u_] += 0.1 * other_rating

            R_f[~np.isnan(train_data)] = train_data[~np.isnan(train_data)]
            print(np.sqrt(np.nanmean((test_data - R_f)**2)))

    if args.submit:
        print("Making submission")

        test_data = pd.read_csv(args.data_dir / "test.csv")
        test_pairs = np.array([[user_map[u], item_map[b]] for u, b in zip(test_data['user_id'], test_data['book_id'])])

        sample_submission = pd.DataFrame(test_data["id"], columns=["id"])

        if args.content_user:
            R_f = 0.9 * R_
            for u_ in tqdm(range(len(R_f))):
                most_similar_users = np.argsort(all_scores[u_])[-301:-1]
                other_rating = np.sum(R_[most_similar_users, :] * all_scores[u_, most_similar_users][:, np.newaxis], axis=0) / np.sum(all_scores[u_, most_similar_users])
                R_f[u_] += 0.1 * other_rating
            R_f[~np.isnan(train_data)] = train_data[~np.isnan(train_data)]

            sample_submission["rating"] = R_f[test_pairs[:, 0], test_pairs[:, 1]]
        else:
            sample_submission["rating"] = R_[test_pairs[:, 0], test_pairs[:, 1]]
        sample_submission.to_csv(args.data_dir / 'submission.csv', index=False)



