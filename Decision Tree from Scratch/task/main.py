import numpy as np


def gini(d: np.ndarray) -> float:
    return 1 - sum((np.bincount(d) / len(d)) ** 2)


def weighted_gini(d1: np.ndarray, d2: np.ndarray) -> float:
    return round((len(d1) * gini(d1) + len(d2) * gini(d2)) / (len(d1) + len(d2)), 5)


def main():
    data, split1, split2 = (np.array([int(x) for x in input().split()]) for _ in range(3))
    print(f'{gini(data):.2f} {weighted_gini(split1, split2):.2f}')


if __name__ == '__main__':
    main()
