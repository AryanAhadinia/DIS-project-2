import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import wandb
from src.matrix_factorization.pq import PQ
from utils import get_train_test

from src.bm25.bm25 import BM25

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--token_dir", type=Path, default = "../../data/tokenized_data")
    parser.add_argument("--output_dir", type=Path, default = "../../data")
    parser.add_argument("--data_dir", type=Path, default="../../data")
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--P_lambda", type=float, default=0.494)
    parser.add_argument("--Q_lambda", type=float, default=0.132)
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--wandb_key", type=str, default="")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--wandb_group", type=str, default="PQ")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    if args.wandb_key != "":
        wandb.login(key=args.wandb_key)
    else:
        wandb.login()
    wandb.init(project="DIS2", group=args.wandb_group, config=vars(args))

    with open(f'{args.token_dir}/tokens.pkl', "rb") as f:
        docs = pickle.load(f)

    with open(f'{args.token_dir}/book_ids.pkl', "rb") as f:
        book_ids = np.array(pickle.load(f))

    lang_params = {
        "k1": 1.4, "b": 0.5
    }

    bm25_ind = BM25(k1=lang_params["k1"], b=lang_params["b"])
    bm25_ind.fit(docs)

    data = pd.read_csv(args.data_dir / "train.csv")

    train_data, test_data = get_train_test(data, args.test)

    if args.skip_train:
        with open(args.output_dir / "pq_model.pkl", "rb") as f:
            pq_model = pickle.load(f)
    else:
        pq_model = PQ(len(data["user_id"].unique()), len(data["book_id"].unique()),
                      d=args.d, lr=args.lr, P_lambda=args.P_lambda, Q_lambda=args.Q_lambda)
        print("Fitting model")
        loss_train, loss_test = pq_model.fit(train_data, args.n_iter, test_data=test_data, log_wandb=True)
        if args.output_dir is not None:
            # Save model with pickle
            with open(args.output_dir / "pq_model.pkl", "wb") as f:
                pickle.dump(pq_model, f)

    R = pq_model.P @ pq_model.Q.T
    # replace the known values with the real values
    R[~np.isnan(train_data)] = train_data[~np.isnan(train_data)]

    user_id = data['user_id'].unique()
    item_id = data['book_id'].unique()

    user_map = {u: i for i, u in enumerate(user_id)}
    item_map = {b: j for j, b in enumerate(item_id)}

    book_id_to_index = {book_id: i for i, book_id in enumerate(book_ids)}
    for k_ in range(args.k):
        R_ = R.copy()
        for b_id in tqdm(item_id, total=len(item_id)):
            if b_id not in book_id_to_index:
                continue
            inds_, scores_ = bm25_ind.match(docs[book_id_to_index[b_id]], k=k_)
            inds = []
            scores = []
            for i, s in zip(inds_, scores_):
                if book_ids[i] in item_map:
                    inds.append(item_map[book_ids[i]])
                    scores.append(s)
            scores = np.array(scores) / np.sum(scores)
            R_[:, item_map[b_id]] = np.sum(R[:, inds] * scores, axis=1)
        R_[~np.isnan(train_data)] = train_data[~np.isnan(train_data)]

        print(k_, np.sqrt(np.nanmean((test_data - R_)**2)))

    if args.submit:

        # user_id = data['user_id'].unique()
        # item_id = data['book_id'].unique()
        test_data = pd.read_csv(args.data_dir / "test.csv")
        test_pairs = np.array([[user_map[u], item_map[b]] for u, b in zip(test_data['user_id'], test_data['book_id'])])

        # user_map = {u: i for i, u in enumerate(user_id)}
        # item_map = {b: j for j, b in enumerate(item_id)}
        #
        # with open(args.output_dir / "model.pkl", "rb") as f:
        #     pq_model = pickle.load(f)
        # predictions = pq_model.predict(test_pairs)

        sample_submission = pd.read_csv(args.data_dir / "sample_submission.csv")
        sample_submission["rating"] = R_[test_pairs[:, 0], test_pairs[:, 1]]
        sample_submission.to_csv(args.data_dir / 'submission.csv', index=False)



