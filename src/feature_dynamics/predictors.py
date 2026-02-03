"""Predictors for SAE feature evolution."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class TokenOnlyPredictor:
    """Token-only baseline: x_{t+1} ≈ B u_t.

    Predicts next SAE features based only on current token embedding.
    """

    def __init__(self, alpha: float = 1.0, n_features: int = None):
        """Initialize token-only predictor.

        Args:
            alpha: Ridge regularization parameter
            n_features: Number of SAE features
        """
        self.alpha = alpha
        self.n_features = n_features
        self.embed_dim = None
        self.embed_matrix = None  # Stored for prediction
        self.models = {}  # Per-feature models if fitting separately
        self.B = None  # Joint model matrix (embed_dim, n_features)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def _prepare_data(self, dataset: List[Dict], embed_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data.

        Args:
            dataset: List of data dictionaries
            embed_matrix: (vocab_size, embed_dim) token embedding matrix

        Returns:
            (X, y) where:
                X: (n_samples, embed_dim) token embeddings
                y: (n_samples, n_features) next SAE activations
        """
        X_list = []
        y_list = []

        for data in dataset:
            sae_acts = data['sae_acts']  # (seq_len, n_features)
            token_ids = data['token_ids']  # (seq_len,)

            # Create pairs (embed(token_t), x_{t+1})
            for t in range(len(token_ids) - 1):
                token_embed = embed_matrix[token_ids[t]]
                X_list.append(token_embed)
                y_list.append(sae_acts[t + 1])

        X = np.array(X_list)  # (n_samples, embed_dim)
        y = np.array(y_list)  # (n_samples, n_features)

        return X, y

    def fit(self, dataset: List[Dict], embed_matrix: np.ndarray, per_feature: bool = False):
        """Fit token-only predictor.

        Args:
            dataset: Training dataset
            embed_matrix: (vocab_size, embed_dim) token embedding matrix
            per_feature: If True, fit separate model per feature
        """
        # Store embedding matrix for prediction
        self.embed_matrix = embed_matrix
        self.embed_dim = embed_matrix.shape[1]

        # Infer n_features if not set
        if self.n_features is None:
            self.n_features = dataset[0]['sae_acts'].shape[1]

        X, y = self._prepare_data(dataset, embed_matrix)

        # Standardize inputs and targets
        X = self.scaler_x.fit_transform(X)
        y = self.scaler_y.fit_transform(y)

       
            # Fit joint model
        model = Ridge(alpha=self.alpha, fit_intercept=True)
        model.fit(X, y)
        self.B = model.coef_.T  # (embed_dim, n_features)
        self.intercept = model.intercept_

    def predict(self, dataset: List[Dict]) -> List[np.ndarray]:
        """Predict next SAE features for dataset.

        Args:
            dataset: Dataset to predict on

        Returns:
            List of prediction arrays (seq_len-1, n_features) for each sequence
        """
        predictions = []

        for data in dataset:
            token_ids = data['token_ids']
            seq_preds = []

            for t in range(len(token_ids) - 1):
                # Get token embedding
                token_embed = self.embed_matrix[token_ids[t]]
                token_embed_scaled = self.scaler_x.transform(token_embed.reshape(1, -1))[0]

                # Predict
                if self.models:
                    # Per-feature models
                    pred = np.array([
                        self.models[i].predict(token_embed_scaled.reshape(1, -1))[0]
                        for i in range(self.n_features)
                    ])
                else:
                    # Joint model
                    pred = token_embed_scaled @ self.B + self.intercept

                # Inverse transform
                pred = self.scaler_y.inverse_transform(pred.reshape(1, -1))[0]
                seq_preds.append(pred)

            predictions.append(np.array(seq_preds))

        return predictions


class StateTokenPredictor:
    """State+token predictor: x_{t+1} ≈ A x_t + B u_t.

    Predicts next SAE features based on current state and token embedding.
    """

    def __init__(self, alpha: float = 1.0, n_features: int = None):
        """Initialize state+token predictor.

        Args:
            alpha: Ridge regularization parameter
            n_features: Number of SAE features
        """
        self.alpha = alpha
        self.n_features = n_features
        self.embed_dim = None
        self.embed_matrix = None  # Stored for prediction
        self.models = {}  # Per-feature models if fitting separately
        self.A = None  # State transition matrix (n_features, n_features)
        self.B = None  # Token input matrix (embed_dim, n_features)
        self.scaler_state = StandardScaler()
        self.scaler_embed = StandardScaler()
        self.scaler_y = StandardScaler()

    def _prepare_data(self, dataset: List[Dict], embed_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data.

        Args:
            dataset: List of data dictionaries
            embed_matrix: (vocab_size, embed_dim) token embedding matrix

        Returns:
            (X_state, X_embed, y) where:
                X_state: (n_samples, n_features) state features
                X_embed: (n_samples, embed_dim) token embeddings
                y: (n_samples, n_features) next SAE activations
        """
        X_state_list = []
        X_embed_list = []
        y_list = []

        for data in dataset:
            sae_acts = data['sae_acts']  # (seq_len, n_features)
            token_ids = data['token_ids']  # (seq_len,)

            # Create pairs ([x_t, embed(u_t)], x_{t+1})
            for t in range(len(token_ids) - 1):
                X_state_list.append(sae_acts[t])
                X_embed_list.append(embed_matrix[token_ids[t]])
                y_list.append(sae_acts[t + 1])

        X_state = np.array(X_state_list)  # (n_samples, n_features)
        X_embed = np.array(X_embed_list)  # (n_samples, embed_dim)
        y = np.array(y_list)  # (n_samples, n_features)

        return X_state, X_embed, y

    def fit(self, dataset: List[Dict], embed_matrix: np.ndarray, per_feature: bool = False):
        """Fit state+token predictor.

        Args:
            dataset: Training dataset
            embed_matrix: (vocab_size, embed_dim) token embedding matrix
            per_feature: If True, fit separate model per feature
        """
        # Store embedding matrix for prediction
        self.embed_matrix = embed_matrix
        self.embed_dim = embed_matrix.shape[1]

        # Infer n_features if not set
        if self.n_features is None:
            self.n_features = dataset[0]['sae_acts'].shape[1]

        X_state, X_embed, y = self._prepare_data(dataset, embed_matrix)

        # Standardize state and embedding features separately
        X_state = self.scaler_state.fit_transform(X_state)
        X_embed = self.scaler_embed.fit_transform(X_embed)

        # Concatenate
        X = np.concatenate([X_state, X_embed], axis=1)

        y = self.scaler_y.fit_transform(y)

        
        # Fit joint model
        model = Ridge(alpha=self.alpha, fit_intercept=True)
        model.fit(X, y)

        # Extract A and B matrices
        coefs = model.coef_  # (n_features, n_features + embed_dim)
        self.A = coefs[:, :self.n_features].T  # (n_features, n_features)
        self.B = coefs[:, self.n_features:].T  # (embed_dim, n_features)
        self.intercept = model.intercept_

    def predict(self, dataset: List[Dict]) -> List[np.ndarray]:
        """Predict next SAE features for dataset.

        Args:
            dataset: Dataset to predict on

        Returns:
            List of prediction arrays (seq_len-1, n_features) for each sequence
        """
        predictions = []

        for data in dataset:
            sae_acts = data['sae_acts']
            token_ids = data['token_ids']
            seq_preds = []

            for t in range(len(token_ids) - 1):
                # Get token embedding
                token_embed = self.embed_matrix[token_ids[t]]
                token_embed_scaled = self.scaler_embed.transform(token_embed.reshape(1, -1))[0]

                # Current state
                x_t = sae_acts[t]
                x_t_scaled = self.scaler_state.transform(x_t.reshape(1, -1))[0]

                # Concatenate features
                features = np.concatenate([x_t_scaled, token_embed_scaled])

                # Predict
                if self.models:
                    # Per-feature models
                    pred = np.array([
                        self.models[i].predict(features.reshape(1, -1))[0]
                        for i in range(self.n_features)
                    ])
                else:
                    # Joint model: pred = x_t @ A + embed(u_t) @ B + intercept
                    pred = (
                        x_t_scaled @ self.A +
                        token_embed_scaled @ self.B +
                        self.intercept
                    )

                # Inverse transform
                pred = self.scaler_y.inverse_transform(pred.reshape(1, -1))[0]
                seq_preds.append(pred)

            predictions.append(np.array(seq_preds))

        return predictions


def save_predictor(predictor, path: Path):
    """Save predictor to disk."""
    with open(path, 'wb') as f:
        pickle.dump(predictor, f)


def load_predictor(path: Path):
    """Load predictor from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)
