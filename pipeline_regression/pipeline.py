from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import clone

from config import CONFIG, REGRESSORS


def build_pipeline(regressor_name=None):
    reg_name = regressor_name if regressor_name else CONFIG["regressor"][0]

    steps = []

    if CONFIG["use_pca"]:
        steps.append(("pca", PCA(n_components=CONFIG["pca_components"])))

    reg = clone(REGRESSORS[reg_name])

    steps.append(("regressor", reg))

    return Pipeline(steps)
