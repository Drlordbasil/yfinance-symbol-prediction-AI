from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import numpy as np
import yfinance as yf
Optimized Python script:


def fetch_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)['Close']
        return data
    except Exception as e:
        raise Exception(f"Error fetching data: {e}")


def preprocess_data(data):
    try:
        scaler = MinMaxScaler()
        return scaler.fit_transform(data.values.reshape(-1, 1))
    except Exception as e:
        raise Exception(f"Error preprocessing data: {e}")


def convert_to_sequences(data_scaled, sequence_length):
    try:
        X = []
        y = []
        for i in range(len(data_scaled) - sequence_length):
            X.append(data_scaled[i:i + sequence_length])
            y.append(data_scaled[i + sequence_length])
        return np.array(X), np.array(y)
    except Exception as e:
        raise Exception(f"Error converting to sequences: {e}")


def split_dataset(X, y, test_size):
    try:
        return train_test_split(X, y, test_size=test_size, shuffle=False)
    except Exception as e:
        raise Exception(f"Error splitting dataset: {e}")


def build_model(sequence_length):
    try:
        model = keras.Sequential([
            keras.layers.LSTM(64, input_shape=(sequence_length, 1)),
            keras.layers.Dense(1)
        ])
        return model
    except Exception as e:
        raise Exception(f"Error building model: {e}")


def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    try:
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs,
                  batch_size=batch_size, verbose=0)
    except Exception as e:
        raise Exception(f"Error training model: {e}")


def evaluate_model(model, X_val, y_val):
    try:
        y_pred = model.predict(X_val)
        mse = np.mean(np.square(y_pred - y_val))
        return mse
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")


def make_realtime_prediction(model, scaler, sequence_length, symbol, start, end):
    try:
        live_data = fetch_data(symbol, start, end)
        live_data_scaled = scaler.transform(live_data.values.reshape(-1, 1))
        last_sequence = np.array([live_data_scaled[-sequence_length:]])
        return model.predict(last_sequence)
    except Exception as e:
        raise Exception(f"Error making real-time prediction: {e}")


def main():
    try:
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        start_date = '2010-01-01'
        end_date = '2021-01-01'
        sequence_length = 30

        data = [fetch_data(symbol, start_date, end_date) for symbol in symbols]
        data_scaled = [preprocess_data(d) for d in data]
        X, y = convert_to_sequences(
            np.concatenate(data_scaled), sequence_length)
        X_train, X_val, y_train, y_val = split_dataset(X, y, test_size=0.2)

        model = build_model(sequence_length)
        train_model(model, X_train, y_train)

        mse = evaluate_model(model, X_val, y_val)
        print("Mean Squared Error:", mse)

        live_symbol = 'AAPL'
        prediction_start_date = '2021-01-01'
        prediction_end_date = '2021-02-01'
        next_prediction = make_realtime_prediction(
            model, MinMaxScaler(), sequence_length, live_symbol, prediction_start_date, prediction_end_date)
        print("Next prediction:", next_prediction)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
