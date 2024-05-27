import numpy as np
import pandas as pd


def gini(d: np.ndarray) -> float:
    return 1 - sum((np.bincount(d) / len(d)) ** 2)


def weighted_gini(d1: np.ndarray, d2: np.ndarray) -> float:
    return round((len(d1) * gini(d1) + len(d2) * gini(d2)) / (len(d1) + len(d2)), 5)


def split(x: pd.DataFrame, y: pd.Series) -> tuple:
    result = None
    for feat in x.columns:
        for val in x[feat].unique():
            left, right = y[x[feat] == val], y[x[feat] != val]
            gini_ind = weighted_gini(left, right)
            if not result or gini_ind < result[0]:
                result = (gini_ind, feat, val, left.index.tolist(), right.index.tolist())
    return result


def main():
    df = pd.read_csv(input(), index_col=0)
    data, target = df.drop(columns='Survived'), df['Survived']
    print(*split(data, target))


if __name__ == '__main__':
    main()
