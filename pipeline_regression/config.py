from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

CONFIG = {
    "random_seed": 42,

    "train_df_path": "dataset/final_regressor/no_smiles/train.csv",
    "test_df_path": "dataset/final_regressor/no_smiles/val.csv",

    "category_specific": "none",

    # Embedding model
    # Options: regular model names OR "custom".
    # Examples: "all-MiniLM-L6-v2", "all-mpnet-base-v2", "modernBERT", "all-roberta-large-v1"
    # "intfloat/e5-large-v2", "BAAI/bge-large-en-v1.5", "mixedbread-ai/mxbai-embed-large-v1", "hkunlp/instructor-large"
    "embedding_model": "all-mpnet-base-v2",
    # Used when embedding_model == "custom".
    "custom_embedding_path": "fine_tuned_embeddings/roberta-base/best_model",  # e.g. "./fine_tuned_embeddings/complex_model_sbert/best_model"
    # Optional explicit tokenizer path for custom HF models (fallbacks to model path/base checkpoint if None).
    "custom_tokenizer_path": None,
    # Options when embedding_model == "custom": "auto", "sentence_transformer", "hf_auto_model"
    "custom_embedding_backend": "sentence_transformer",

    "target_column": "slm_accuracy_fraction",

    # Dimensionality reduction
    "use_pca": False,
    "pca_components": 300,

    # Regression head selection
    "regressor": ["svr"],

    # Averaging ensemble, used only when more than one regressor is selected.
    "voting_strategy": "average",

    # Save the trained sklearn pipeline/ensemble.
    # "save_model": True,
}

REGRESSORS = {
    "knn": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    "linear": LinearRegression(n_jobs=-1),
    "ridge": Ridge(),
    "rf": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=CONFIG["random_seed"]),
    "svr": SVR(kernel="rbf", verbose=True),
    "dt": DecisionTreeRegressor(random_state=CONFIG["random_seed"]),
    "gb": GradientBoostingRegressor(random_state=CONFIG["random_seed"], verbose=1),
    "xgb": xgb.XGBRegressor(n_jobs=-1, random_state=CONFIG["random_seed"], verbosity=1),
    "lgb": lgb.LGBMRegressor(n_jobs=-1, random_state=CONFIG["random_seed"]),
    "mlp": MLPRegressor(hidden_layer_sizes=(1024), max_iter=500, random_state=CONFIG["random_seed"], verbose=True),
}
