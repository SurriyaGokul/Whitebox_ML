import numpy as np
from typing import Union
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from utils.Node import Node

class DecisionTree:
    """
    A simple Decision Tree implementation written purely using numpy and pandas supporting both classification and regression tasks.
    Parameters
    ----------
    categorical : bool, default=False
        Whether to treat all features as categorical.
    task : str, default="Classification"
        The type of task to perform: "Classification" or "Regression".
    metric : str, default="entropy"
        The impurity metric to use for classification: "entropy" or "gini".
    max_depth : int, optional
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain fewer than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    max_features : int, float, or str, optional
        The number of features to consider when looking for the best split.
    random_state : int, optional
        Controls the randomness of the estimator for reproducibility.
    Attributes
    ----------
    tree : Node or None
        The root node of the fitted decision tree.
    Methods
    -------
    entropy(y)
        Compute the entropy of a label array y.
    gini(y)
        Compute the Gini impurity of a label array y.
    calc_mse(y)
        Compute the mean squared error of a target array y.
    split(x, y)
        Find the best feature and value to split the data for the current node.
    fit(x, y, depth=0)
        Build the decision tree from the training set (x, y).
    predict(x)
        Predict target values for samples in x.
    Notes
    -----
    - Supports both numerical and categorical features.
    - Handles both classification and regression tasks.
    - Uses recursive binary splitting based on the chosen impurity metric.
    """
    
    def __init__(
        self,
        categorical: bool = False,
        task: str = "Classification",
        metric: str = "entropy",
        max_depth: int = None,
        min_samples_split: int = 2,
        max_features: Union[int, float, str] = None,
        random_state: int = None,
    ):
        self.task = task
        self.max_depth = max_depth
        self.metric = metric
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None
        self.categorical = categorical
        if self.random_state is not None:
            np.random.seed(self.random_state)

    @staticmethod
    def entropy(y):
        # simple entropy computation
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-9))  # add epsilon to avoid log(0)

    @staticmethod
    def gini(y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)

    @staticmethod
    def calc_mse(y):
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def split(self, x, y):
        df = x.copy()
        y = np.array(y)
        best_min = float("inf")
        best_split = None
        for col in df.columns:
            mixed = pd.concat([df[col], pd.DataFrame(y, columns=["y"])], axis=1)
            mixed = mixed.sort_values(by=col)

            if pd.api.types.is_numeric_dtype(df[col]):
                # numeric feature
                col_min, split = float("inf"), None
                values = mixed[col].values
                for i in range(len(values) - 1):
                    mid = (values[i] + values[i + 1]) / 2
                    left_y = mixed[mixed[col] <= mid]["y"]
                    right_y = mixed[mixed[col] > mid]["y"]
                    if self.task.lower()=="classification":
                        if self.metric == "entropy":
                            impurity = (
                                len(left_y) / len(y) * self.entropy(left_y)
                                + len(right_y) / len(y) * self.entropy(right_y)
                            )
                        elif self.metric == "gini":
                            impurity = (
                                len(left_y) / len(y) * self.gini(left_y)
                                + len(right_y) / len(y) * self.gini(right_y)
                            )
                        else:
                            raise ValueError("Unknown metric")
                    elif self.task.lower()=="regression":
                        impurity = (
                                len(left_y) / len(y) * self.calc_mse(left_y)
                                + len(right_y) / len(y) * self.calc_mse(right_y)
                            )
                    else:
                        raise ValueError("Unknown task")
                    
                    if impurity < col_min:
                        col_min = impurity
                        split = mid

                if col_min < best_min:
                    best_min = col_min
                    best_split = (col, split, "numerical")

            else:
                col_min, split = float("inf"), None
                categories = mixed[col].unique()
                for category in categories:
                    left_y = mixed[mixed[col] == category]["y"]
                    right_y = mixed[mixed[col] != category]["y"]
                    if self.task.lower()=="classification":
                        if self.metric == "entropy":
                            impurity = (
                                len(left_y) / len(y) * self.entropy(left_y)
                                + len(right_y) / len(y) * self.entropy(right_y)
                            )
                        elif self.metric == "gini":
                            impurity = (
                                len(left_y) / len(y) * self.gini(left_y)
                                + len(right_y) / len(y) * self.gini(right_y)
                            )
                        else:
                            raise ValueError("Unknown metric")
                    elif self.task.lower()=="regression":
                        impurity = (
                                len(left_y) / len(y) * self.calc_mse(left_y)
                                + len(right_y) / len(y) * self.calc_mse(right_y)
                            )
                    else:
                        raise ValueError("Unknown task")
                    
                    if impurity < col_min:
                        col_min = impurity
                        split = category

                if col_min < best_min:
                    best_min = col_min
                    best_split = (col, split, "categorical")

        return {
            "feature": best_split[0],
            "split_value": best_split[1],
            "type": best_split[2],
            "impurity": best_min,
        }
    
    def fit(self, x, y, depth = 0):

        if len(np.unique(y)) == 1:
            node = Node(prediction=y.iloc[0])
            return node
        
        df = x.copy()
        res = self.split(df, y)        
        node = Node(
            feature=res["feature"],
            split_value=res["split_value"],
            node_type=res["type"],
            impurity=res["impurity"]
        )

        left = pd.DataFrame(df[df[res["feature"]] <= res["split_value"] if res["type"] == "numerical" else df[res["feature"]] == res["split_value"]])
        right = pd.DataFrame(df[df[res["feature"]] > res["split_value"] if res["type"] == "numerical" else df[res["feature"]] != res["split_value"]])
        
        left_y = y[left.index]
        right_y = y[right.index]
        if len(left_y) < self.min_samples_split or len(right_y) < self.min_samples_split or (self.max_depth is not None and self.max_depth<=depth):
            if self.task == "regression":
                node.prediction = np.mean(y)
            else:
                values, counts = np.unique(y, return_counts=True)
                node.prediction = values[np.argmax(counts)]
            return node
        
        node.left = self.fit(left, left_y, depth+1)
        node.right = self.fit(right, right_y, depth+1)

        if depth==0:
            self.tree = node

        return node

    def predict(self, x: pd.DataFrame):
        if self.tree is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        
        def traverse_tree(node, sample):
            if node.left is None and node.right is None:
                return node.prediction
            if node.node_type == "numerical":
                if sample[node.feature] <= node.split_value:
                    return traverse_tree(node.left, sample)
                else:
                    return traverse_tree(node.right, sample)
            else:
                if sample[node.feature]==node.split_value:
                    return traverse_tree(node.left,sample)
                else:
                    return traverse_tree(node.right,sample)
        
        results = []
        for _, row in x.iterrows():
            results.append(traverse_tree(self.tree, row))
        return results
    
    