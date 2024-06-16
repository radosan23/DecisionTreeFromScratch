import numpy as np
import pandas as pd


class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.term = False
        self.label = None
        self.feature = None
        self.value = None

    def set_split(self, feature, value) -> None:
        self.feature = feature
        self.value = value

    def set_term(self, label) -> None:
        self.term = True
        self.label = label


class DecisionTree:
    def __init__(self, min_samples=1):
        self.root = Node()
        self.min_samples = min_samples

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self._recursive_split(self.root, x, y)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return x.apply(self._recursive_predict, args=(self.root, ), axis=1).to_numpy()

    @staticmethod
    def _gini(d: np.ndarray | pd.Series) -> float:
        return 1 - sum((np.bincount(d) / len(d)) ** 2)

    def _weighted_gini(self, d1: np.ndarray | pd.Series, d2: np.ndarray | pd.Series) -> float:
        return round((len(d1) * self._gini(d1) + len(d2) * self._gini(d2)) / (len(d1) + len(d2)), 5)

    def _split(self, x: pd.DataFrame, y: pd.Series) -> tuple:
        result = None
        for feat in x.columns:
            for val in x[feat].unique():
                if x[feat].dtype == float:
                    left, right = y[x[feat] <= val], y[x[feat] > val]
                else:
                    left, right = y[x[feat] == val], y[x[feat] != val]
                gini_ind = self._weighted_gini(left, right)
                if not result or gini_ind < result[0]:
                    result = (gini_ind, feat, val, left.index.tolist(), right.index.tolist())
        return result

    def _recursive_split(self, node: Node, x: pd.DataFrame, y: pd.Series) -> None:
        if (x.shape[0] <= self.min_samples) or (x.value_counts().shape[0] == 1) or (self._gini(y) == 0):
            node.set_term(y.mode()[0])
            return
        _, feat, val, left_ind, right_ind = self._split(x, y)
        node.set_split(feat, val)
        node.left, node.right = Node(), Node()
        self._recursive_split(node.left, x.iloc[left_ind].reset_index(drop=True),
                              y.iloc[left_ind].reset_index(drop=True))
        self._recursive_split(node.right, x.iloc[right_ind].reset_index(drop=True),
                              y.iloc[right_ind].reset_index(drop=True))

    def _recursive_predict(self, sample: pd.Series, node: Node) -> int:
        if node.term:
            return node.label
        next_node = node.left if sample[node.feature] == node.value else node.right
        return self._recursive_predict(sample, next_node)


def main():
    df = pd.read_csv(input(), index_col=0)
    x_train, y_train = df.drop(columns='Survived'), df['Survived']

    tree = DecisionTree(min_samples=74)
    print(*[x.round(3) if type(x) is float else x for x in tree._split(x_train, y_train)])


if __name__ == '__main__':
    main()
