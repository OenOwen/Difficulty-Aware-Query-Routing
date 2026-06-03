from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

CONFIG = {
    "random_seed": 42,

    "train_df_path": "dataset/final_binary/train.csv",
    "test_df_path": "dataset/final_binary/val.csv",

    "category_specific": "none",

    # Embedding model
    # Options: regular model names OR "custom".
    # Examples: "all-MiniLM-L6-v2", "all-mpnet-base-v2", "modernBERT", "all-roberta-large-v1"
    # "intfloat/e5-large-v2", "BAAI/bge-large-en-v1.5", "mixedbread-ai/mxbai-embed-large-v1", "hkunlp/instructor-large"
    "embedding_model": "all-roberta-large-v1",
    # Used when embedding_model == "custom".
    "custom_embedding_path": "fine_tuned_embeddings/roberta-base/best_model",  # e.g. "./fine_tuned_embeddings/complex_model_sbert/best_model"
    # Optional explicit tokenizer path for custom HF models (fallbacks to model path/base checkpoint if None).
    "custom_tokenizer_path": None,
    # Options when embedding_model == "custom": "auto", "sentence_transformer", "hf_auto_model"
    "custom_embedding_backend": "sentence_transformer",

    "vote_accuracy_threshold": 0.75,

    # Dimensionality reduction
    "use_pca": False,
    "pca_components": 300,

    # Classifier selection
    "classifier": ["svm"],

    # "hard" or "soft" vote
    "voting_strategy": "soft",

    # Balancing strategy
    # Options: "none", "augment", "class_weight"
    "balance_strategy": "none",
    "minority_label": 0,
    "augment_factor": 2,
    "augment_noise_std": 0.0005,


    # "save_model": False,
}

CLASSIFIERS = {
    "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "logreg": LogisticRegression(max_iter=100, n_jobs=-1),
    "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1),
    "svm": SVC(kernel='poly', probability=True, verbose=True),
    "dt": DecisionTreeClassifier(),
    "nb": GaussianNB(),
    "gb": GradientBoostingClassifier(),
    "xgb": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
    "lgb": lgb.LGBMClassifier(n_jobs=-1)
}
