"""Predictors for SAE feature evolution."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class TokenOnlyPredictor:
    """Token-only baseline: x_{t+1} ≈ B u_t.

    Predicts next SAE features based only on current token.
    """

    def __init__(self, alpha: float = 1.0, n_features: int = None, vocab_size: int = None):
        """Initialize token-only predictor.

        Args:
            alpha: Ridge regularization parameter
            n_features: Number of SAE features
            vocab_size: Vocabulary size
        """
        self.alpha = alpha
        self.n_features = n_features
        self.vocab_size = vocab_size
        self.models = {}  # Per-feature models if fitting separately
        self.B = None  # Joint model matrix (vocab_size, n_features)
        self.scaler_y = StandardScaler()

    def _prepare_data(self, dataset: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data.

        Args:
            dataset: List of data dictionaries

        Returns:
            (X, y) where:
                X: (n_samples, vocab_size) one-hot encoded tokens
                y: (n_samples, n_features) next SAE activations
        """
        X_list = []
        y_list = []

        for data in dataset:
            sae_acts = data['sae_acts']  # (seq_len, n_features)
            token_ids = data['token_ids']  # (seq_len,)

            # Create pairs (token_t, x_{t+1})
            for t in range(len(token_ids) - 1):
                # One-hot encode token
                token_onehot = np.zeros(self.vocab_size)
                token_onehot[token_ids[t]] = 1.0

                X_list.append(token_onehot)
                y_list.append(sae_acts[t + 1])

        X = np.array(X_list)  # (n_samples, vocab_size)
        y = np.array(y_list)  # (n_samples, n_features)

        return X, y

    def fit(self, dataset: List[Dict], per_feature: bool = False):
        """Fit token-only predictor.

        Args:
            dataset: Training dataset
            per_feature: If True, fit separate model per feature
        """
        # Infer dimensions if not set
        if self.n_features is None:
            self.n_features = dataset[0]['sae_acts'].shape[1]
        if self.vocab_size is None:
            # Get max token ID across dataset
            max_token = max(data['token_ids'].max() for data in dataset)
            self.vocab_size = int(max_token) + 1

        X, y = self._prepare_data(dataset)

        # Standardize targets
        y = self.scaler_y.fit_transform(y)

        if per_feature:
            # Fit separate model for each feature
            self.models = {}
            for i in range(self.n_features):
                model = Ridge(alpha=self.alpha, fit_intercept=True)
                model.fit(X, y[:, i])
                self.models[i] = model
        else:
            # Fit joint model
            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(X, y)
            self.B = model.coef_.T  # (vocab_size, n_features)
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
                # One-hot encode token
                token_onehot = np.zeros(self.vocab_size)
                token_onehot[token_ids[t]] = 1.0

                # Predict
                if self.models:
                    # Per-feature models
                    pred = np.array([
                        self.models[i].predict(token_onehot.reshape(1, -1))[0]
                        for i in range(self.n_features)
                    ])
                else:
                    # Joint model
                    pred = token_onehot @ self.B + self.intercept

                # Inverse transform
                pred = self.scaler_y.inverse_transform(pred.reshape(1, -1))[0]
                seq_preds.append(pred)

            predictions.append(np.array(seq_preds))

        return predictions


class StateTokenPredictor:
    """State+token predictor: x_{t+1} ≈ A x_t + B u_t.

    Predicts next SAE features based on current state and token.
    """

    def __init__(self, alpha: float = 1.0, n_features: int = None, vocab_size: int = None):
        """Initialize state+token predictor.

        Args:
            alpha: Ridge regularization parameter
            n_features: Number of SAE features
            vocab_size: Vocabulary size
        """
        self.alpha = alpha
        self.n_features = n_features
        self.vocab_size = vocab_size
        self.models = {}  # Per-feature models if fitting separately
        self.A = None  # State transition matrix (n_features, n_features)
        self.B = None  # Token input matrix (vocab_size, n_features)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def _prepare_data(self, dataset: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data.

        Args:
            dataset: List of data dictionaries

        Returns:
            (X, y) where:
                X: (n_samples, n_features + vocab_size) concatenated [x_t, u_t]
                y: (n_samples, n_features) next SAE activations
        """
        X_list = []
        y_list = []

        for data in dataset:
            sae_acts = data['sae_acts']  # (seq_len, n_features)
            token_ids = data['token_ids']  # (seq_len,)

            # Create pairs ([x_t, u_t], x_{t+1})
            for t in range(len(token_ids) - 1):
                # One-hot encode token
                token_onehot = np.zeros(self.vocab_size)
                token_onehot[token_ids[t]] = 1.0

                # Concatenate state and token
                x_t = sae_acts[t]
                features = np.concatenate([x_t, token_onehot])

                X_list.append(features)
                y_list.append(sae_acts[t + 1])

        X = np.array(X_list)  # (n_samples, n_features + vocab_size)
        y = np.array(y_list)  # (n_samples, n_features)

        return X, y

    def fit(self, dataset: List[Dict], per_feature: bool = False):
        """Fit state+token predictor.

        Args:
            dataset: Training dataset
            per_feature: If True, fit separate model per feature
        """
        # Infer dimensions if not set
        if self.n_features is None:
            self.n_features = dataset[0]['sae_acts'].shape[1]
        if self.vocab_size is None:
            max_token = max(data['token_ids'].max() for data in dataset)
            self.vocab_size = int(max_token) + 1

        X, y = self._prepare_data(dataset)

        # Standardize inputs and targets
        # Only standardize the state features, not the one-hot token features
        X_state = X[:, :self.n_features]
        X_token = X[:, self.n_features:]

        X_state = self.scaler_x.fit_transform(X_state)
        X = np.concatenate([X_state, X_token], axis=1)

        y = self.scaler_y.fit_transform(y)

        if per_feature:
            # Fit separate model for each feature
            self.models = {}
            for i in range(self.n_features):
                model = Ridge(alpha=self.alpha, fit_intercept=True)
                model.fit(X, y[:, i])
                self.models[i] = model
        else:
            # Fit joint model
            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(X, y)

            # Extract A and B matrices
            coefs = model.coef_  # (n_features, n_features + vocab_size)
            self.A = coefs[:, :self.n_features].T  # (n_features, n_features)
            self.B = coefs[:, self.n_features:].T  # (vocab_size, n_features)
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
                # One-hot encode token
                token_onehot = np.zeros(self.vocab_size)
                token_onehot[token_ids[t]] = 1.0

                # Current state
                x_t = sae_acts[t]
                x_t_scaled = self.scaler_x.transform(x_t.reshape(1, -1))[0]

                # Concatenate features
                features = np.concatenate([x_t_scaled, token_onehot])

                # Predict
                if self.models:
                    # Per-feature models
                    pred = np.array([
                        self.models[i].predict(features.reshape(1, -1))[0]
                        for i in range(self.n_features)
                    ])
                else:
                    # Joint model: pred = A @ x_t + B @ u_t + intercept
                    pred = (
                        self.A @ x_t_scaled +
                        self.B @ token_onehot +
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
