import numpy as np
import wandb
from tqdm import tqdm


class PQ:
    """
    Model for matrix factorization using gradient descent. 
    The model is a simple matrix factorization model with a user matrix P and an item matrix Q.
    The loss function is the squared error between the observed values and the predicted values. 

    Attributes:
        P (np.ndarray): User matrix.
        Q (np.ndarray): Item matrix.
        lr (float): Learning rate for gradient descent.
        P_lambda (float): Regularization parameter for user matrix.
        Q_lambda (float): Regularization parameter for item matrix.

    Methods:
        fit(data, num_iterations, test_data=None, log_wandb=False) -> Tuple[List[float], List[float]]: Fit the model to the data.
        compute_loss(data) -> float: Compute the loss of the model.
        abs_error(data) -> float: Compute the absolute error of the model.
        predict(user_item_pairs) -> np.ndarray: Predict the values for user-item pairs.
    """
    def __init__(self, num_users, num_items, d, lr=0.01, P_lambda=0.1, Q_lambda=0.1):
        """
        Initialize the model.
        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            d (int): Dimension of the latent space.
            lr (float): Learning rate for gradient descent.
            P_lambda (float): Regularization parameter for user matrix.
            Q_lambda (float): Regularization parameter for item matrix.
        """
        self.P = np.random.rand(num_users, d)
        self.Q = np.random.rand(num_items, d)
        self.lr = lr
        self.P_lambda = P_lambda
        self.Q_lambda = Q_lambda

    def fit(self, data, num_iterations, test_data=None, log_wandb=False):
        """
        Fit the model to the data.
        Args:
            data (np.ndarray): Matrix of observed values for training.
            num_iterations (int): Number of iterations for gradient descent.
            test_data (np.ndarray): Matrix of observed values for testing.
            log_wandb (bool): Whether to log results to wandb.  
        Returns:
            Tuple[List[float], List[float]]: Losses for training and testing.
        """
        num_users, num_items = data.shape
        E = np.full((num_users, num_items), np.nan) # Error matrix
        user_item_indices = np.argwhere(~np.isnan(data)) # Indices of observed values
        L = []
        L_test = []
        for n in tqdm(range(num_iterations)):
            for i, j in user_item_indices:
                E[i, j] = data[i, j] - np.dot(self.P[i, :], self.Q[j, :].T) # Compute error
                gradient_P = 2 * E[i, j] * self.Q[j, :] - self.P_lambda * self.P[i, :] # Compute gradient for user matrix
                gradient_Q = 2 * E[i, j] * self.P[i, :] - self.Q_lambda * self.Q[j, :] # Compute gradient for item matrix
                self.P[i, :] += self.lr * np.clip(gradient_P, -1e3, 1e3)
                self.Q[j, :] += self.lr * np.clip(gradient_Q, -1e3, 1e3)
            
            loss = self.compute_loss(data) # Compute loss
            L.append(loss)
            test_loss = None
            if test_data is not None:
                test_loss = self.compute_loss(test_data) # Compute test loss
                L_test.append(test_loss)
            if log_wandb: # Log results to wandb
                train_abs_error = self.abs_error(data)
                if test_data is None:
                    result_dict = {'step': int(n), 'train/loss': float(loss), 'train/abs_error': float(train_abs_error)}
                else:
                    result_dict = {'step': int(n), 'train/loss': float(loss),
                                   'eval/loss': float(test_loss), 'train/abs_error': float(train_abs_error),
                                      'eval/abs_error': float(self.abs_error(test_data))}
                wandb.log(result_dict)

        return L, L_test

    def compute_loss(self, data):
        """
        Compute the loss of the model (root mean squared error).
        Args:
            data (np.ndarray): Matrix of observed values.
        Returns:
            float: Loss of the model.
        """
        return np.sqrt(np.nanmean((data - np.dot(self.P, self.Q.T))**2))

    def abs_error(self, data):
        """
        Compute the absolute error of the model.
        Args:
            data (np.ndarray): Matrix of observed values.
        Returns:
            float: Absolute error of the model.
        """
        return np.nanmean(np.abs(data - np.dot(self.P, self.Q.T)))

    def predict(self, user_item_pairs):
        """
        Predict the values for user-item pairs.
        Args:
            user_item_pairs (np.ndarray): Matrix of user-item pairs.
        Returns:
            np.ndarray: Predicted values.
        """
        return np.sum(self.P[user_item_pairs[:, 0], :] * self.Q[user_item_pairs[:, 1], :], axis=1)

