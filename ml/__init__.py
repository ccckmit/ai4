"""ml - Machine Learning toolkit (sklearn-style, NumPy-based).

Submodules
----------
linear_models : LinearRegression, LogisticRegression
tree          : DecisionTree
ensemble      : RandomForest
clustering    : KMeans
decomposition : PCA
metrics       : accuracy_score, mean_squared_error, etc.
preprocessing : StandardScaler, train_test_split
"""

from .linear_models import LinearRegression, LogisticRegression
from .tree import DecisionTree
from .ensemble import RandomForest
from .clustering import KMeans
from .decomposition import PCA
from .preprocessing import StandardScaler, train_test_split
from .metrics import accuracy_score, mean_squared_error, r2_score

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "KMeans",
    "PCA",
    "StandardScaler",
    "train_test_split",
    "accuracy_score",
    "mean_squared_error",
    "r2_score",
]