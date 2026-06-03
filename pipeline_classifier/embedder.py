import torch
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


class Embedder:
    def __init__(
        self,
        model_name: str,
        custom_model_path: str | None = None,
        custom_backend: str = "auto",
        custom_tokenizer_path: str | None = None,
    ):
        """
        Initialize the embedder and load the model.
        """
        self.model_name = model_name

        # Device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
        else:
            self.device = "cpu"

        # Load model
        if model_name.lower() == "custom":
            if not custom_model_path:
                raise ValueError("CONFIG['custom_embedding_path'] must be set when embedding_model='custom'")

            backend = custom_backend.lower()
            if backend not in {"auto", "sentence_transformer", "hf_auto_model"}:
                raise ValueError("custom_backend must be 'auto', 'sentence_transformer', or 'hf_auto_model'")

            if backend in {"auto", "sentence_transformer"}:
                try:
                    self.model = SentenceTransformer(custom_model_path, device=self.device)
                    self.is_modernbert = False
                    return
                except Exception:
                    if backend == "sentence_transformer":
                        raise

            tokenizer_candidates = []
            if custom_tokenizer_path:
                tokenizer_candidates.append(custom_tokenizer_path)
            tokenizer_candidates.append(custom_model_path)

            config_path = os.path.join(custom_model_path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    base_name = cfg.get("_name_or_path")
                    if base_name and base_name not in tokenizer_candidates:
                        tokenizer_candidates.append(base_name)
                except Exception:
                    pass

            last_error = None
            self.tokenizer = None
            for path in tokenizer_candidates:
                for use_fast in (True, False):
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=use_fast)
                        break
                    except Exception as err:
                        last_error = err
                if self.tokenizer is not None:
                    break

            if self.tokenizer is None:
                raise ValueError(
                    "Failed to load tokenizer for custom embedding model. "
                    "Set CONFIG['custom_tokenizer_path'] to a compatible tokenizer "
                    "or install required tokenizer deps (e.g. sentencepiece/tiktoken)."
                ) from last_error

            self.model = AutoModel.from_pretrained(custom_model_path).to(self.device)
            self.is_modernbert = True
        elif model_name.lower() == "modernbert":
            self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
            self.model = AutoModel.from_pretrained("answerdotai/ModernBERT-base").to(self.device)
            self.is_modernbert = True
        else:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.is_modernbert = False

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.is_modernbert:
            self.model.eval()
            embeddings = []

            with torch.no_grad():
                for text in texts:
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=128
                    ).to(self.device)

                    outputs = self.model(**inputs)
                    mean_emb = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(mean_emb.cpu().numpy()[0])

            return np.array(embeddings)

        return (
            self.model
            .encode(texts, normalize_embeddings=True, convert_to_tensor=True)
            .cpu()
            .numpy()
        )


def augment_minority(embeddings: np.ndarray, labels: np.ndarray, minority_label=0, factor=2, noise_std=0.01, random_seed=42):
    if factor < 2:
        return embeddings, labels

    np.random.seed(random_seed)

    X_min = embeddings[labels == minority_label]
    y_min = labels[labels == minority_label]
    if len(X_min) == 0:
        return embeddings, labels

    augmented_X = [X_min]
    augmented_y = [y_min]

    for _ in range(factor - 1):
        noise = np.random.normal(0, noise_std, X_min.shape)
        augmented_X.append(X_min + noise)
        augmented_y.append(y_min)

    X_aug = np.vstack(augmented_X)
    y_aug = np.hstack(augmented_y)

    X_major = embeddings[labels != minority_label]
    y_major = labels[labels != minority_label]

    embeddings = np.vstack([X_aug, X_major])
    labels = np.hstack([y_aug, y_major])

    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    return embeddings[indices], labels[indices]
