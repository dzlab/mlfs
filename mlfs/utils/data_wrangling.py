import numpy as np

class DataWrangling():
    """ Data Wrangling.
    Args:
        alpha (float): The learning rate.

    Returns:
        bool: The return value. True for success, False otherwise.
    """
    def __init__(self, alpha=0.1, batch=False):
        self.alpha = alpha

    def normalize(self, X):
        """Normalize the features in X
        """
        mu = X.mean()
        sigma = X.std()
        normalizer = lambda row: (row - mu) / sigma
        X_norm = X.apply(normalizer, axis=1)
        return X_norm, mu, sigma

    def shuffle(self, X, y):
        """Randomly shuffle the rows of the dataset
        """
        #df = sk.utils.shuffle(df)
        m = np.shape(X)[0]
        indexes = np.arange(m)
        np.random.shuffle(indexes)
        X_shufl = X.reindex(indexes)
        y_shufl = y.reindex(indexes)
        return X_shufl, y_shufl

    def split(self, X, y, val_size=0.0, test_ratio=0.4):
        """Split the dataset into a training and testing sets
        """
        m = np.shape(X)[0]
        test_size = int(m * test_ratio)
        train_size = m - test_size
        return X[ :train_size], X[train_size: ], y[ :train_size], y[train_size: ]
