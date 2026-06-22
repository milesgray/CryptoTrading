import numpy as np
from typing import Dict, Any, List
from encoder import RetrievalServiceEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class RetrievalForecaster:
    def __init__(self, encoder_service: RetrievalServiceEncoder):
        self.encoder_service = encoder_service
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        """Build an LSTM model for forecasting."""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(60, 56)),  # 60 timesteps, 56 features
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(1)  # Predict next price
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def forecast(
        self,
        prices: np.ndarray,
        order_book: Dict[str, Any],
        k: int = 5
    ) -> Dict[str, Any]:
        """Forecast future prices using retrieval-augmented LSTM."""
        # Retrieve similar segments
        retrieved = self.encoder_service.retrieve_segments(prices, order_book, k=k)
        
        # Prepare input for LSTM (reshape prices to 60 timesteps with 56 features)
        X = np.array([self.encoder_service.encode_segment(prices[i:i+1], order_book) for i in range(60)])
        X = X.reshape((1, 60, 56))  # Reshape for LSTM
        
        # Predict
        prediction = self.model.predict(X, verbose=0)[0][0]
        
        return {
            "retrieved": retrieved,
            "prediction": float(prediction)
        }