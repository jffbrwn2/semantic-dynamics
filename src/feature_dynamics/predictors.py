"""Predictors for SAE feature evolution."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_act_key(use_pre_relu: bool) -> str:
    """Get the activation key based on use_pre_relu flag."""
    return 'pre_relu' if use_pre_relu else 'sae_acts'


class TokenOnlyPredictor:
    """Token-only baseline: x_{t+1} ≈ B u_t.

    Predicts next SAE features based only on current token embedding.
    """

    def __init__(self, alpha: float = 1.0, n_features: int = None, use_pre_relu: bool = True):
        """Initialize token-only predictor.

        Args:
            alpha: Ridge regularization parameter
            n_features: Number of SAE features
            use_pre_relu: If True, use pre-ReLU activations instead of sae_acts
        """
        self.alpha = alpha
        self.n_features = n_features
        self.use_pre_relu = use_pre_relu
        self.act_key = get_act_key(use_pre_relu)
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
                y: (n_samples, n_features) next activations
        """
        X_list = []
        y_list = []

        for data in dataset:
            acts = data[self.act_key]  # (seq_len, n_features)
            token_ids = data['token_ids']  # (seq_len,)

            # Create pairs (embed(token_t), x_{t+1})
            for t in range(len(token_ids) - 1):
                token_embed = embed_matrix[token_ids[t]]
                X_list.append(token_embed)
                y_list.append(acts[t + 1])

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
            self.n_features = dataset[0][self.act_key].shape[1]

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

    def simulate_trajectory(self, data: Dict) -> np.ndarray:
        """Simulate trajectory using only token inputs.

        For token-only model, this is just applying B*u_t for each step.
        The first element is the true initial state for comparison.

        Args:
            data: Single data dictionary with token_ids

        Returns:
            Predicted trajectory (seq_len, n_features)
        """
        acts = data[self.act_key]
        token_ids = data['token_ids']
        seq_len = len(token_ids)

        # Start with true initial state
        trajectory = [acts[0]]

        for t in range(seq_len - 1):
            # Get token embedding
            token_embed = self.embed_matrix[token_ids[t]]
            token_embed_scaled = self.scaler_x.transform(token_embed.reshape(1, -1))[0]

            # Predict
            pred_scaled = token_embed_scaled @ self.B + self.intercept

            # Inverse transform
            pred = self.scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
            trajectory.append(pred)

        return np.array(trajectory)

    # Generation interface methods
    def reset(self, initial_features: np.ndarray = None):
        """Reset predictor state. TokenOnly has no state to reset."""
        pass

    def predict_next(self, token_embed: np.ndarray, previous_features: np.ndarray = None) -> np.ndarray:
        """Predict next features from token embedding.

        Args:
            token_embed: Token embedding vector (embed_dim,)
            previous_features: Ignored for TokenOnly predictor

        Returns:
            Predicted features (n_features,)
        """
        token_embed_scaled = self.scaler_x.transform(token_embed.reshape(1, -1))[0]
        pred_scaled = token_embed_scaled @ self.B + self.intercept
        pred = self.scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
        return pred


class StateTokenPredictor:
    """State+token predictor: x_{t+1} ≈ A x_t + B u_t.

    Predicts next SAE features based on current state and token embedding.
    """

    def __init__(self, alpha: float = 1.0, n_features: int = None, use_pre_relu: bool = True):
        """Initialize state+token predictor.

        Args:
            alpha: Ridge regularization parameter
            n_features: Number of SAE features
            use_pre_relu: If True, use pre-ReLU activations instead of sae_acts
        """
        self.alpha = alpha
        self.n_features = n_features
        self.use_pre_relu = use_pre_relu
        self.act_key = get_act_key(use_pre_relu)
        self.embed_dim = None
        self.embed_matrix = None  # Stored for prediction
        self.models = {}  # Per-feature models if fitting separately
        self.A = None  # State transition matrix (n_features, n_features)
        self.B = None  # Token input matrix (embed_dim, n_features)
        self.scaler_state = StandardScaler()
        self.scaler_embed = StandardScaler()
        self.scaler_y = StandardScaler()
        self._current_state = None  # For generation interface

    def _prepare_data(self, dataset: List[Dict], embed_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data.

        Args:
            dataset: List of data dictionaries
            embed_matrix: (vocab_size, embed_dim) token embedding matrix

        Returns:
            (X_state, X_embed, y) where:
                X_state: (n_samples, n_features) state features
                X_embed: (n_samples, embed_dim) token embeddings
                y: (n_samples, n_features) next activations
        """
        X_state_list = []
        X_embed_list = []
        y_list = []

        for data in dataset:
            acts = data[self.act_key]  # (seq_len, n_features)
            token_ids = data['token_ids']  # (seq_len,)

            # Create pairs ([x_t, embed(u_t)], x_{t+1})
            for t in range(len(token_ids) - 1):
                X_state_list.append(acts[t])
                X_embed_list.append(embed_matrix[token_ids[t]])
                y_list.append(acts[t + 1])

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
            self.n_features = dataset[0][self.act_key].shape[1]

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
        """Predict next activations for dataset using true state at each step.

        Args:
            dataset: Dataset to predict on

        Returns:
            List of prediction arrays (seq_len-1, n_features) for each sequence
        """
        predictions = []

        for data in dataset:
            acts = data[self.act_key]
            token_ids = data['token_ids']
            seq_preds = []

            for t in range(len(token_ids) - 1):
                # Get token embedding
                token_embed = self.embed_matrix[token_ids[t]]
                token_embed_scaled = self.scaler_embed.transform(token_embed.reshape(1, -1))[0]

                # Current state
                x_t = acts[t]
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

    def simulate_trajectory(self, data: Dict) -> np.ndarray:
        """Simulate trajectory by unrolling predictions from initial state.

        Instead of using true state at each step, uses predicted state.
        This gives the "autonomous" dynamics: x_{t+1} = A * x_hat_t + B * u_t

        Args:
            data: Single data dictionary with token_ids and initial state

        Returns:
            Predicted trajectory (seq_len, n_features) starting from x_0
        """
        acts = data[self.act_key]
        token_ids = data['token_ids']
        seq_len = len(token_ids)

        # Initialize with true initial state
        trajectory = [acts[0]]
        x_hat = acts[0].copy()

        for t in range(seq_len - 1):
            # Get token embedding
            token_embed = self.embed_matrix[token_ids[t]]
            token_embed_scaled = self.scaler_embed.transform(token_embed.reshape(1, -1))[0]

            # Scale current predicted state
            x_hat_scaled = self.scaler_state.transform(x_hat.reshape(1, -1))[0]

            # Predict next state
            pred_scaled = (
                x_hat_scaled @ self.A +
                token_embed_scaled @ self.B +
                self.intercept
            )

            # Inverse transform
            x_hat = self.scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
            trajectory.append(x_hat.copy())

        return np.array(trajectory)

    # Generation interface methods
    def reset(self, initial_features: np.ndarray = None):
        """Reset predictor state.

        Args:
            initial_features: Initial SAE features to use as state
        """
        self._current_state = initial_features.copy() if initial_features is not None else None

    def predict_next(self, token_embed: np.ndarray, previous_features: np.ndarray = None) -> np.ndarray:
        """Predict next features from token embedding and previous features.

        Args:
            token_embed: Token embedding vector (embed_dim,)
            previous_features: Previous SAE features (n_features,)
                             If None, uses internal _current_state

        Returns:
            Predicted features (n_features,)
        """
        # Use provided features or internal state
        if previous_features is None:
            previous_features = self._current_state
        if previous_features is None:
            raise ValueError("previous_features must be provided or reset() must be called first")

        # Scale inputs
        token_embed_scaled = self.scaler_embed.transform(token_embed.reshape(1, -1))[0]
        state_scaled = self.scaler_state.transform(previous_features.reshape(1, -1))[0]

        # Predict
        pred_scaled = (
            state_scaled @ self.A +
            token_embed_scaled @ self.B +
            self.intercept
        )

        # Inverse transform
        pred = self.scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]

        # Update internal state
        self._current_state = pred.copy()

        return pred


class SequenceDataset(Dataset):
    """Dataset for sequence prediction."""

    def __init__(self, sequences: List[Tuple[np.ndarray, np.ndarray]]):
        """Initialize dataset.

        Args:
            sequences: List of (input_seq, target_seq) tuples
                input_seq: (seq_len, input_dim)
                target_seq: (seq_len, output_dim)
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        inputs, targets = self.sequences[idx]
        return torch.FloatTensor(inputs), torch.FloatTensor(targets)


class LSTMModel(nn.Module):
    """LSTM model for sequence prediction."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_dim)
        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out: (batch, seq_len, hidden_dim)
        output = self.fc(lstm_out)
        # output: (batch, seq_len, output_dim)
        return output, hidden


class RNNTokenOnlyPredictor:
    """RNN-based token-only predictor: h_t, x_{t+1} = RNN(u_t, h_{t-1}).

    Uses only token embeddings as input to predict next pre_relu state.
    The RNN maintains an internal hidden state for temporal context.
    """

    def __init__(
        self,
        n_features: int = None,
        use_pre_relu: bool = True,
        hidden_size: int = None,
        num_layers: int = 1,
        learning_rate: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """Initialize RNN token-only predictor.

        Args:
            n_features: Number of SAE features
            use_pre_relu: If True, use pre-ReLU activations
            hidden_size: RNN hidden state size (defaults to n_features if None)
            num_layers: Number of RNN layers
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to run on (cpu/cuda)
        """
        self.n_features = n_features
        self.use_pre_relu = use_pre_relu
        self.act_key = get_act_key(use_pre_relu)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.embed_dim = None
        self.embed_matrix = None
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self._hidden = None  # For generation interface

    def _build_model(self):
        """Build LSTM model."""
        if self.hidden_size is None:
            self.hidden_size = self.n_features

        self.model = LSTMModel(
            self.embed_dim,
            self.hidden_size,
            self.n_features,
            self.num_layers
        ).to(self.device)

    def _prepare_sequences(self, dataset: List[Dict], embed_matrix: np.ndarray):
        """Prepare sequences for training.

        Args:
            dataset: List of data dictionaries
            embed_matrix: Token embedding matrix

        Returns:
            List of (input_seq, target_seq) tuples
        """
        sequences = []

        for data in dataset:
            acts = data[self.act_key]  # (seq_len, n_features)
            token_ids = data['token_ids']  # (seq_len,)

            if len(token_ids) < 2:
                continue

            # Input: token embeddings at t=0..T-2
            # Target: activations at t=1..T-1
            input_seq = np.array([embed_matrix[tid] for tid in token_ids[:-1]])
            target_seq = acts[1:]

            sequences.append((input_seq, target_seq))

        return sequences

    def fit(self, dataset: List[Dict], embed_matrix: np.ndarray, per_feature: bool = False,
            val_dataset: List[Dict] = None):
        """Fit RNN token-only predictor.

        Args:
            dataset: Training dataset
            embed_matrix: Token embedding matrix
            per_feature: Ignored (kept for API compatibility)
            val_dataset: Optional validation dataset
        """
        self.embed_matrix = embed_matrix
        self.embed_dim = embed_matrix.shape[1]

        if self.n_features is None:
            self.n_features = dataset[0][self.act_key].shape[1]

        # Prepare sequences
        sequences = self._prepare_sequences(dataset, embed_matrix)

        # Fit scalers on concatenated data
        all_inputs = np.concatenate([seq[0] for seq in sequences], axis=0)
        all_targets = np.concatenate([seq[1] for seq in sequences], axis=0)
        self.scaler_x.fit(all_inputs)
        self.scaler_y.fit(all_targets)

        # Scale sequences
        scaled_sequences = [
            (self.scaler_x.transform(inp), self.scaler_y.transform(tgt))
            for inp, tgt in sequences
        ]

        # Prepare validation data if provided
        val_loader = None
        if val_dataset is not None:
            val_sequences = self._prepare_sequences(val_dataset, embed_matrix)
            val_scaled_sequences = [
                (self.scaler_x.transform(inp), self.scaler_y.transform(tgt))
                for inp, tgt in val_sequences
            ]
            val_dataset_obj = SequenceDataset(val_scaled_sequences)
            val_loader = DataLoader(
                val_dataset_obj,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self._collate_fn
            )

        # Build model
        self._build_model()

        # Create dataloader
        train_dataset = SequenceDataset(scaled_sequences)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            total_loss = 0
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs, _ = self.model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_inputs, batch_targets in val_loader:
                        batch_inputs = batch_inputs.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        outputs, _ = self.model(batch_inputs)
                        loss = criterion(outputs, batch_targets)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)

                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}")

    def _collate_fn(self, batch):
        """Collate function for variable length sequences."""
        # For now, just batch sequences (they should be similar length)
        inputs = torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True)
        return inputs, targets

    def predict(self, dataset: List[Dict]) -> List[np.ndarray]:
        """Predict next activations for dataset.

        Args:
            dataset: Dataset to predict on

        Returns:
            List of prediction arrays for each sequence
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for data in dataset:
                token_ids = data['token_ids']

                if len(token_ids) < 2:
                    predictions.append(np.array([]))
                    continue

                # Prepare input sequence
                input_seq = np.array([self.embed_matrix[tid] for tid in token_ids[:-1]])
                input_seq_scaled = self.scaler_x.transform(input_seq)
                input_tensor = torch.FloatTensor(input_seq_scaled).unsqueeze(0).to(self.device)

                # Predict
                outputs, _ = self.model(input_tensor)
                outputs_np = outputs.squeeze(0).cpu().numpy()

                # Inverse transform
                predictions_unscaled = self.scaler_y.inverse_transform(outputs_np)
                predictions.append(predictions_unscaled)

        return predictions

    def simulate_trajectory(self, data: Dict) -> np.ndarray:
        """Simulate trajectory using only token inputs.

        Args:
            data: Single data dictionary

        Returns:
            Predicted trajectory (seq_len, n_features)
        """
        # For RNN, this is same as predict since it only uses tokens
        acts = data[self.act_key]
        token_ids = data['token_ids']

        trajectory = [acts[0]]  # Start with true initial state

        if len(token_ids) < 2:
            return np.array(trajectory)

        # Get predictions
        pred = self.predict([data])[0]
        trajectory.extend(pred)

        return np.array(trajectory)

    # Generation interface methods
    def reset(self, initial_features: np.ndarray = None):
        """Reset RNN hidden state to None."""
        self._hidden = None

    def predict_next(self, token_embed: np.ndarray, previous_features: np.ndarray = None) -> np.ndarray:
        """Predict next features from token embedding.

        Args:
            token_embed: Token embedding vector (embed_dim,)
            previous_features: Ignored for RNN TokenOnly predictor

        Returns:
            Predicted features (n_features,)
        """
        self.model.eval()

        # Scale input
        token_embed_scaled = self.scaler_x.transform(token_embed.reshape(1, -1))
        input_tensor = torch.FloatTensor(token_embed_scaled).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Predict with current hidden state
            output, self._hidden = self.model(input_tensor, self._hidden)
            output_np = output.squeeze(0).squeeze(0).cpu().numpy()

        # Inverse transform
        pred = self.scaler_y.inverse_transform(output_np.reshape(1, -1))[0]

        return pred


class RNNStateTokenPredictor:
    """RNN-based state+token predictor: h_t, x_{t+1} = RNN([x_t, u_t], h_{t-1}).

    Uses both current state and token embeddings to predict next state.
    """

    def __init__(
        self,
        n_features: int = None,
        use_pre_relu: bool = True,
        hidden_size: int = None,
        num_layers: int = 1,
        learning_rate: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """Initialize RNN state+token predictor.

        Args:
            n_features: Number of SAE features
            use_pre_relu: If True, use pre-ReLU activations
            hidden_size: RNN hidden state size (defaults to n_features if None)
            num_layers: Number of RNN layers
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to run on (cpu/cuda)
        """
        self.n_features = n_features
        self.use_pre_relu = use_pre_relu
        self.act_key = get_act_key(use_pre_relu)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.embed_dim = None
        self.embed_matrix = None
        self.model = None
        self.scaler_state = StandardScaler()
        self.scaler_embed = StandardScaler()
        self.scaler_y = StandardScaler()
        self._hidden = None  # For generation interface
        self._current_state = None  # For generation interface

    def _build_model(self):
        """Build LSTM model."""
        if self.hidden_size is None:
            self.hidden_size = self.n_features

        input_dim = self.n_features + self.embed_dim

        self.model = LSTMModel(
            input_dim,
            self.hidden_size,
            self.n_features,
            self.num_layers
        ).to(self.device)

    def _prepare_sequences(self, dataset: List[Dict], embed_matrix: np.ndarray):
        """Prepare sequences for training.

        Args:
            dataset: List of data dictionaries
            embed_matrix: Token embedding matrix

        Returns:
            List of (input_seq, target_seq) tuples
        """
        sequences = []

        for data in dataset:
            acts = data[self.act_key]  # (seq_len, n_features)
            token_ids = data['token_ids']  # (seq_len,)

            if len(token_ids) < 2:
                continue

            # Input: [x_t, u_t] at t=0..T-2
            # Target: x_{t+1} at t=1..T-1
            input_states = acts[:-1]
            input_embeds = np.array([embed_matrix[tid] for tid in token_ids[:-1]])
            target_seq = acts[1:]

            sequences.append((input_states, input_embeds, target_seq))

        return sequences

    def fit(self, dataset: List[Dict], embed_matrix: np.ndarray, per_feature: bool = False,
            val_dataset: List[Dict] = None):
        """Fit RNN state+token predictor.

        Args:
            dataset: Training dataset
            embed_matrix: Token embedding matrix
            per_feature: Ignored (kept for API compatibility)
            val_dataset: Optional validation dataset
        """
        self.embed_matrix = embed_matrix
        self.embed_dim = embed_matrix.shape[1]

        if self.n_features is None:
            self.n_features = dataset[0][self.act_key].shape[1]

        # Prepare sequences
        sequences = self._prepare_sequences(dataset, embed_matrix)

        # Fit scalers
        all_states = np.concatenate([seq[0] for seq in sequences], axis=0)
        all_embeds = np.concatenate([seq[1] for seq in sequences], axis=0)
        all_targets = np.concatenate([seq[2] for seq in sequences], axis=0)

        self.scaler_state.fit(all_states)
        self.scaler_embed.fit(all_embeds)
        self.scaler_y.fit(all_targets)

        # Scale and concatenate sequences
        scaled_sequences = []
        for states, embeds, targets in sequences:
            states_scaled = self.scaler_state.transform(states)
            embeds_scaled = self.scaler_embed.transform(embeds)
            inputs = np.concatenate([states_scaled, embeds_scaled], axis=1)
            targets_scaled = self.scaler_y.transform(targets)
            scaled_sequences.append((inputs, targets_scaled))

        # Prepare validation data if provided
        val_loader = None
        if val_dataset is not None:
            val_sequences = self._prepare_sequences(val_dataset, embed_matrix)
            val_scaled_sequences = []
            for states, embeds, targets in val_sequences:
                states_scaled = self.scaler_state.transform(states)
                embeds_scaled = self.scaler_embed.transform(embeds)
                inputs = np.concatenate([states_scaled, embeds_scaled], axis=1)
                targets_scaled = self.scaler_y.transform(targets)
                val_scaled_sequences.append((inputs, targets_scaled))

            val_dataset_obj = SequenceDataset(val_scaled_sequences)
            val_loader = DataLoader(
                val_dataset_obj,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self._collate_fn
            )

        # Build model
        self._build_model()

        # Create dataloader
        train_dataset = SequenceDataset(scaled_sequences)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            total_loss = 0
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs, _ = self.model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_inputs, batch_targets in val_loader:
                        batch_inputs = batch_inputs.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        outputs, _ = self.model(batch_inputs)
                        loss = criterion(outputs, batch_targets)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)

                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}")

    def _collate_fn(self, batch):
        """Collate function for variable length sequences."""
        inputs = torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True)
        return inputs, targets

    def predict(self, dataset: List[Dict]) -> List[np.ndarray]:
        """Predict next activations using true state at each step.

        Args:
            dataset: Dataset to predict on

        Returns:
            List of prediction arrays for each sequence
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for data in dataset:
                acts = data[self.act_key]
                token_ids = data['token_ids']

                if len(token_ids) < 2:
                    predictions.append(np.array([]))
                    continue

                # Prepare input sequence
                input_states = acts[:-1]
                input_embeds = np.array([self.embed_matrix[tid] for tid in token_ids[:-1]])

                states_scaled = self.scaler_state.transform(input_states)
                embeds_scaled = self.scaler_embed.transform(input_embeds)
                inputs = np.concatenate([states_scaled, embeds_scaled], axis=1)

                input_tensor = torch.FloatTensor(inputs).unsqueeze(0).to(self.device)

                # Predict
                outputs, _ = self.model(input_tensor)
                outputs_np = outputs.squeeze(0).cpu().numpy()

                # Inverse transform
                predictions_unscaled = self.scaler_y.inverse_transform(outputs_np)
                predictions.append(predictions_unscaled)

        return predictions

    def simulate_trajectory(self, data: Dict) -> np.ndarray:
        """Simulate trajectory by unrolling predictions from initial state.

        Args:
            data: Single data dictionary

        Returns:
            Predicted trajectory (seq_len, n_features)
        """
        self.model.eval()

        acts = data[self.act_key]
        token_ids = data['token_ids']
        seq_len = len(token_ids)

        trajectory = [acts[0]]  # True initial state
        x_hat = acts[0].copy()

        with torch.no_grad():
            hidden = None
            for t in range(seq_len - 1):
                # Get token embedding
                token_embed = self.embed_matrix[token_ids[t]]
                token_embed_scaled = self.scaler_embed.transform(token_embed.reshape(1, -1))[0]

                # Scale current predicted state
                x_hat_scaled = self.scaler_state.transform(x_hat.reshape(1, -1))[0]

                # Concatenate inputs
                inputs = np.concatenate([x_hat_scaled, token_embed_scaled])
                input_tensor = torch.FloatTensor(inputs).unsqueeze(0).unsqueeze(0).to(self.device)

                # Predict next state
                output, hidden = self.model(input_tensor, hidden)
                output_np = output.squeeze(0).squeeze(0).cpu().numpy()

                # Inverse transform
                x_hat = self.scaler_y.inverse_transform(output_np.reshape(1, -1))[0]
                trajectory.append(x_hat.copy())

        return np.array(trajectory)

    # Generation interface methods
    def reset(self, initial_features: np.ndarray = None):
        """Reset RNN hidden state and SAE feature state.

        Args:
            initial_features: Initial SAE features to use as state
        """
        self._hidden = None
        self._current_state = initial_features.copy() if initial_features is not None else None

    def predict_next(self, token_embed: np.ndarray, previous_features: np.ndarray = None) -> np.ndarray:
        """Predict next features from token embedding and previous features.

        Args:
            token_embed: Token embedding vector (embed_dim,)
            previous_features: Previous SAE features (n_features,)
                             If None, uses internal _current_state

        Returns:
            Predicted features (n_features,)
        """
        # Use provided features or internal state
        if previous_features is None:
            previous_features = self._current_state
        if previous_features is None:
            raise ValueError("previous_features must be provided or reset() must be called first")

        self.model.eval()

        # Scale inputs
        token_embed_scaled = self.scaler_embed.transform(token_embed.reshape(1, -1))[0]
        state_scaled = self.scaler_state.transform(previous_features.reshape(1, -1))[0]

        # Concatenate
        inputs = np.concatenate([state_scaled, token_embed_scaled])
        input_tensor = torch.FloatTensor(inputs).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Predict with current hidden state
            output, self._hidden = self.model(input_tensor, self._hidden)
            output_np = output.squeeze(0).squeeze(0).cpu().numpy()

        # Inverse transform
        pred = self.scaler_y.inverse_transform(output_np.reshape(1, -1))[0]

        # Update internal state
        self._current_state = pred.copy()

        return pred


def save_predictor(predictor, path: Path):
    """Save predictor to disk."""
    with open(path, 'wb') as f:
        pickle.dump(predictor, f)


def load_predictor(path: Path):
    """Load predictor from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)
