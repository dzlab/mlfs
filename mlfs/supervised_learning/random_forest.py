import numpy as np
import panda as pd

class RandomForest():
    """Random Forest regressor
    Args:
        n_trees (integer): number of trees in this forest
        samples_sz (integer): number of samples for each tree
        min_leaf (integer): minimum number of samples in a leaf node

    """
    def __init__(self, n_trees=10, samples_sz=None, min_leaf=1, max_features=1, max_depth=None, bootstrap=False):
        self.n_trees, self.samples_sz, self.min_leaf = n_trees, samples_sz, min_leaf
        self.bootstrap = bootstrap
        self.trees = [DecitionTree(min_leaf, max_features) for i in range(n_trees)]
    
    @property
    def value(self):
        """return the value associated with RandomForest"""
        return np.mean([t.value for t in self.trees])
    
    def fit(self, X, y):
        m = X.shape[0]
        # select samples with bootstrapping or not
        indices = np.random.choice(m, size=self.samples_sz, replace=self.bootstrap)
        for t in self.trees:
            t.fit(X.iloc[indices], y[indices], indices)
        
    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees])

class DecistionTree():
    """Decision tree
    """
    def __init__(self, min_leaf=1, max_features=1.0):
        self.min_leaf, self.max_features = min_leaf, max_features
        self.value = float("Inf")
        self.score = float("Inf")
        self.split, self.right, self.left = None, None, None

    def _find_split(self, indices, columns):
        """ Find the best split cell by look at every columns and every cell
        """
        X = self.X
        y = self.y
        for c in columns:
            column = X[c]
            # iterate over all samples (or bootstrapped ones) and check
            for i in indices:
                # we cannot access to the row like this as e.g. 0
                split_cell = column[i]
                rindex = column > split_cell
                lindex = column <= split_cell
                if column[rindex].sum()==0: continue
                score = y[rindex].sum() * y[rindex].std() + y[lindex].sum() * y[lindex].std()
                if score < self.score:
                    self.score = score
                    self.split = c
                    self.cell = split_cell
                    self.rindex = rindex
                    self.lindex = lindex

    def fit(self, X, y, indices=None):
        # init parameters
        self.X, self.y, self.indices = X, y, indices
        self.samples = X.shape[0]
        self.value = np.mean(y)
        # return if minumm required samples is reached
        if self.samples <= self.min_leaf: return
        # do fit
        indices = indices if('indices' in dir()) else range(m)
        columns = np.random.choice(X.columns, size=(len(X.columns)*self.max_features))
        # look for the best feature to split on
        self._find_split(indices, columns)
        # make the split and create more trees
        if self.split != None:
            # build right sub-tree
            self.right = DecitionTree(min_leaf=self.min_leaf, max_features=self.max_features)
            Xr = X.loc[self.rindex]
            self.right.fit(Xr, y[self.rindex], Xr.index)
            # build left sub-tree
            self.left = DecitionTree(min_leaf=self.min_leaf, max_features=self.max_features)
            lindex = self.lindex
            Xl = X.loc[self.lindex]
            self.left.fit(Xl, y[lindex], Xl.index)

    def predict(self, X):
        """Run prediction"""
        pass

    @property
    def is_leaf(self):
        return self.score == float("Inf")

    def __repr__(self):
        return f'samples={self.samples} value={self.value} score={self.score} split={self.split}'
