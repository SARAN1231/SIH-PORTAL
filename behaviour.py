# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load your dataset (replace 'your_data.csv' with your dataset file)
data = pd.read_csv('your_data.csv')

# Preprocess the data
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
data['Value'] = scaler.fit_transform(data['Value'].values.reshape(-1, 1))

# Create sequences for training
sequence_length = 10  # Adjust this value based on your data
sequences = []
for i in range(len(data) - sequence_length + 1):
    sequences.append(data['Value'].values[i:i+sequence_length])

# Convert sequences to NumPy array
sequences = np.array(sequences)

# Split the dataset into training and testing
train_size = int(len(sequences) * 0.8)
train_data = sequences[:train_size]
test_data = sequences[train_size:]

# Create X_train and y_train
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# Reshape the data for LSTM input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error (MSE) as a measure of anomaly
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Define a threshold for anomaly detection
threshold = 0.1  # Adjust this threshold based on your data

# Detect anomalies
anomalies = np.where(y_test > threshold, 1, 0)

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.plot(anomalies, 'ro', markersize=4, label='Anomaly')
plt.legend()
plt.title('Anomaly Detection using LSTM')
plt.show()
