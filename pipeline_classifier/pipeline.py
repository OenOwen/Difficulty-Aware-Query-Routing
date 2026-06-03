from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import clone

from config import CONFIG, CLASSIFIERS


def build_pipeline(classifier_name=None):
    clf_name = classifier_name if classifier_name else CONFIG["classifier"]

    steps = []

    if CONFIG["use_pca"]:
        steps.append(("pca", PCA(n_components=CONFIG["pca_components"])))

    clf = clone(CLASSIFIERS[clf_name])

    if CONFIG.get("balance_strategy") == "class_weight":
        if "class_weight" in clf.get_params():
            clf.set_params(class_weight="balanced")

    steps.append(("classifier", clf))

    return Pipeline(steps)