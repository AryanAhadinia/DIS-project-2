import numpy as np
import wandb
from tqdm import tqdm


class PQ:
    def __init__(self, num_users, num_items, d, lr=0.01, P_lambda=0.1, Q_lambda=0.1):
        self.P = np.random.rand(num_users, d)
        self.Q = np.random.rand(num_items, d)
        self.lr = lr
        self.P_lambda = P_lambda
        self.Q_lambda = Q_lambda

    def fit(self, data, num_iterations, test_data=None, log_wandb=False):
        num_users, num_items = data.shape
        E = np.full((num_users, num_items), np.nan)
        user_item_indices = np.argwhere(~np.isnan(data))
        L = []
        L_test = []
        for n in tqdm(range(num_iterations)):
            grads_P = np.zeros_like(self.P)
            grads_Q = np.zeros_like(self.Q)
            for i, j in user_item_indices:
                E[i, j] = data[i, j] - np.dot(self.P[i, :], self.Q[j, :].T)
                gradient_P = 2 * E[i, j] * self.Q[j, :] - self.P_lambda * self.P[i, :]
                gradient_Q = 2 * E[i, j] * self.P[i, :] - self.Q_lambda * self.Q[j, :]
                # self.P[i, :] += self.lr * np.clip(gradient_P, -1e3, 1e3)
                # self.Q[j, :] += self.lr * np.clip(gradient_Q, -1e3, 1e3)
                grads_P[i, :] += np.clip(gradient_P, -1e3, 1e3)
                grads_Q[j, :] += np.clip(gradient_Q, -1e3, 1e3)
            self.P += self.lr * grads_P
            self.Q += self.lr * grads_Q
            loss = self.compute_loss(data)
            L.append(loss)
            test_loss = None
            if test_data is not None:
                test_loss = self.compute_loss(test_data)
                L_test.append(test_loss)
            if log_wandb:
                if test_data is None:
                    result_dict = {'step': int(n), 'train/loss': float(loss)}
                else:
                    result_dict = {'step': int(n), 'train/loss': float(loss),
                                   'eval/loss': float(test_loss)}
                wandb.log(result_dict)

        return L, L_test

    def compute_loss(self, data):
        return np.sqrt(np.nanmean((data - np.dot(self.P, self.Q.T))**2))

    def predict(self, user_item_pairs):
        return np.sum(self.P[user_item_pairs[:, 0], :] * self.Q[user_item_pairs[:, 1], :], axis=1)

