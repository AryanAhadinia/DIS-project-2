import numpy as np
import pandas as pd
import ray
from ray import tune
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

all_user_ids = np.unique(
    np.concatenate([train_df["user_id"].unique(), test_df["user_id"].unique()])
)
all_user_ids.sort()
all_book_ids = np.unique(
    np.concatenate([train_df["book_id"].unique(), test_df["book_id"].unique()])
)
all_book_ids.sort()

user_id_to_index = {user_id: i for i, user_id in enumerate(all_user_ids)}
book_id_to_index = {book_id: i for i, book_id in enumerate(all_book_ids)}

num_users = len(all_user_ids)
num_items = len(all_book_ids)

train_data_triplets = []
for index, row in train_df.iterrows():
    user_index = user_id_to_index[row["user_id"]]
    book_index = book_id_to_index[row["book_id"]]
    rating = row["rating"]
    train_data_triplets.append((user_index, book_index, rating))

test_data_triplets = []
for index, row in test_df.iterrows():
    user_index = user_id_to_index[row["user_id"]]
    book_index = book_id_to_index[row["book_id"]]
    test_data_triplets.append((user_index, book_index))

train_data_array = np.array(train_data_triplets)
train_data_array


class RatingDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        u, i, r = self.triplets[idx]
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(r, dtype=torch.float),
        )


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=8, mlp_layers=[16, 8], dropout=0.2):
        super(NeuMF, self).__init__()

        self.user_embedding_gmf = nn.Embedding(num_users, mf_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, mf_dim)

        self.user_embedding_mlp = nn.Embedding(num_users, mlp_layers[0] // 2)
        self.item_embedding_mlp = nn.Embedding(num_items, mlp_layers[0] // 2)

        mlp_modules = []
        input_size = mlp_layers[0]
        for layer_size in mlp_layers[1:]:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))  # Add Dropout layer
            input_size = layer_size
        self.mlp = nn.Sequential(*mlp_modules)

        predict_size = mf_dim + mlp_layers[-1]
        self.predict_layer = nn.Linear(predict_size, 1)

        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

    def forward(self, user, item):
        gmf_u = self.user_embedding_gmf(user)
        gmf_i = self.item_embedding_gmf(item)
        gmf_out = gmf_u * gmf_i

        mlp_u = self.user_embedding_mlp(user)
        mlp_i = self.item_embedding_mlp(item)
        mlp_cat = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_out = self.mlp(mlp_cat)

        cat_out = torch.cat([gmf_out, mlp_out], dim=-1)
        pred = self.predict_layer(cat_out)
        return pred.squeeze()


def train_one_fold(
    train_data,
    val_data,
    num_users,
    num_items,
    mf_dim,
    mlp_layers,
    dropout,
    lr,
    weight_decay,
    batch_size,
    epochs=10,
    patience=3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")
    model = NeuMF(
        num_users, num_items, mf_dim=mf_dim, mlp_layers=mlp_layers, dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_dataset = RatingDataset(train_data)
    val_dataset = RatingDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_rmse = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            optimizer.zero_grad()
            pred = model(u, i)
            loss = loss_fn(pred, r)
            loss.backward()
            optimizer.step()

        model.eval()
        val_sse = 0.0
        val_samples = 0
        with torch.no_grad():
            for u, i, r in val_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                pred = model(u, i)
                val_sse += ((pred - r) ** 2).sum().item()
                val_samples += len(r)
        val_mse = val_sse / val_samples
        val_rmse = val_mse**0.5

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_val_rmse


def k_fold_cv(
    all_data,
    num_users,
    num_items,
    mf_dim,
    mlp_layers,
    dropout,
    lr,
    weight_decay,
    batch_size,
    k,
    epochs,
    patience,
):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_rmse_list = []
    for train_idx, val_idx in kf.split(all_data):
        train_data = all_data[train_idx]
        val_data = all_data[val_idx]
        fold_rmse = train_one_fold(
            train_data,
            val_data,
            num_users,
            num_items,
            mf_dim,
            mlp_layers,
            dropout,
            lr,
            weight_decay,
            batch_size,
            epochs,
            patience,
        )
        fold_rmse_list.append(fold_rmse)
        break  # Only train one fold for now
    return np.mean(fold_rmse_list), np.std(fold_rmse_list)


def train_with_k_fold(config):
    mf_dim = config["mf_dim"]
    mlp_layers = config["mlp_layers"]
    dropout = config["dropout"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    batch_size = config["batch_size"]

    val_rmse, std = k_fold_cv(
        train_data_array,
        num_users,
        num_items,
        mf_dim,
        mlp_layers,
        dropout,
        lr,
        weight_decay,
        batch_size,
        k=5,
        epochs=200,
        patience=5,
    )

    return {"val_rmse": val_rmse, "std": std}


ray.init()

search_space = {
    "mf_dim": tune.grid_search([4, 8, 16]),
    "mlp_layers": tune.grid_search(
        [[8, 4], [16, 4], [16, 8], [32, 16, 4], [32, 16, 8]]
    ),
    "dropout": tune.grid_search([0.1, 0.15, 0.20, 0.25]),
    "lr": tune.grid_search([1e-2, 5e-3, 1e-3]),
    "weight_decay": tune.grid_search([1e-3, 1e-4, 1e-5]),
    "batch_size": tune.grid_search([512, 1024, 2048, 4096]),
}

analysis = tune.run(
    train_with_k_fold,
    config=search_space,
    metric="val_rmse",
    mode="min",
    resources_per_trial={"cpu": 1, "gpu": 0},
)

best_config = analysis.get_best_config(metric="val_rmse", mode="min")
print("Best config found:", best_config)

df = analysis.results_df
df.to_csv("new_mf_tune_results.csv")

ray.shutdown()
