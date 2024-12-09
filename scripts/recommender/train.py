import argparse
from pathlib import Path

import pandas as pd
import wandb

from scripts.recommender.utils import get_train_test
from src.matrix_factorization.pq import PQ

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="../../data")
    parser.add_argument("--output_dir", type=Path, default="")
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--P_lambda", type=float, default=0.1)
    parser.add_argument("--Q_lambda", type=float, default=0.1)
    parser.add_argument("--n_iter", type=int, default=15)
    parser.add_argument("--wandb_key", type=str, default="")
    parser.add_argument("--test", type=bool, default=False)
    args = parser.parse_args()

    if args.wandb_key != "":
        wandb.login(key=args.wandb_key)
    else:
        wandb.login()
    wandb.init(project="DIS2", group=args.wandb_group, config=vars(args))

    data = pd.read_csv(args.data_dir / "train.csv")

    train_data, test_data = get_train_test(data, args.test)

    pq_model = PQ(len(data["user_id"].unique()), len(data["item_id"].unique()),
                  d=args.d, lr=args.lr, P_lambda=args.P_lambda, Q_lambda=args.Q_lambda)

    loss_train, loss_test = pq_model.fit(train_data, args.n_iter, test_data=test_data)

    if args.output_dir != "":
        pq_model.P.to_csv(args.output_dir / "P.csv")
        pq_model.Q.to_csv(args.output_dir / "Q.csv")
        pd.DataFrame(loss_train).to_csv(args.output_dir / "loss_train.csv")
        pd.DataFrame(loss_test).to_csv(args.output_dir / "loss_test.csv")
