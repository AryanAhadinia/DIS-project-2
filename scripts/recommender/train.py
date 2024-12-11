import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

from utils import get_train_test
from src.matrix_factorization.pq import PQ

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="../../data")
    parser.add_argument("--output_dir", type=Path, default="")
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--P_lambda", type=float, default=0.25)
    parser.add_argument("--Q_lambda", type=float, default=0.07)
    parser.add_argument("--n_iter", type=int, default=15)
    parser.add_argument("--wandb_key", type=str, default="")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--wandb_group", type=str, default="PQ")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if args.wandb_key != "":
        wandb.login(key=args.wandb_key)
    else:
        wandb.login()
    wandb.init(project="DIS2", group=args.wandb_group, config=vars(args))

    data = pd.read_csv(args.data_dir / "train.csv")

    train_data, test_data = get_train_test(data, args.test)

    pq_model = PQ(len(data["user_id"].unique()), len(data["book_id"].unique()),
                  d=args.d, lr=args.lr, P_lambda=args.P_lambda, Q_lambda=args.Q_lambda)

    print("Fitting model")
    loss_train, loss_test = pq_model.fit(train_data, args.n_iter, test_data=test_data, log_wandb=True)

    if args.output_dir is not None:
        # Save model with pickle
        with open(args.output_dir / "model.pkl", "wb") as f:
            pickle.dump(pq_model, f)

    if args.submit:

        user_id = data['user_id'].unique()
        item_id = data['book_id'].unique()

        user_map = {u: i for i, u in enumerate(user_id)}
        item_map = {b: j for j, b in enumerate(item_id)}


        with open(args.output_dir / "model.pkl", "rb") as f:
            pq_model = pickle.load(f)
        test_data = pd.read_csv(args.data_dir / "test.csv")
        # test_data, _ = get_train_test(test_data, False)
        test_pairs = np.array([[user_map[u], item_map[b]] for u, b in zip(test_data['user_id'], test_data['book_id'])])
        # print(test_pairs)
        predictions = pq_model.predict(test_pairs)

        sample_submission = pd.read_csv(args.data_dir / "sample_submission.csv")
        sample_submission["rating"] = predictions.reshape(-1).tolist()
        sample_submission.to_csv(args.data_dir / 'submission.csv', index=False)
