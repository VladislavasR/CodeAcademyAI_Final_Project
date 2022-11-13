import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from LSTM_model import LSTM_adam_model
from GRU_model import GRU_adam_model


# Company stock to predict
print('Enter company short name:')
company_id = input()
if company_id == "":
    company_id = "ADSK"

# PARAMETERS
prediction_days = 60 # Choose prediction days timeframe
epochs = 100 # Epochs count

# Timeframe of prices
start = dt.datetime(2012,1,1)
end = dt.datetime(2021,1,1)

# Get stock price data
data = web.DataReader(company_id, 'yahoo', start, end)

# Data preparation
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1,1))


# Create training dataset
x_train = []
y_train = []

for x in range(prediction_days, len(data_scaled)):
    x_train.append(data_scaled[x-prediction_days:x, 0])
    y_train.append(data_scaled[x, 0])

# Convert to numpy
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Get LSTM model
my_lstm_model = LSTM_adam_model(epochs)
lstm_model = my_lstm_model.getModel(x_train, y_train)

# Get GRU model
my_gru_model = GRU_adam_model(epochs)
gru_model = my_gru_model.getModel(x_train, y_train)

# Get test data
start_test = dt.datetime(2021,1,1)
end_test = dt.datetime.now()

data_test = web.DataReader(company_id, 'yahoo', start_test, end_test)
actual_prices = data_test['Close'].values

total_dataset = pd.concat((data['Close'], data_test['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(data_test) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# Get prediction on test data
x_test = []
for i in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[i-prediction_days:i, 0])

x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get predictions from LSTM model
predicted_closing_price_lstm=lstm_model.predict(x_test)
predicted_closing_price_lstm=scaler.inverse_transform(predicted_closing_price_lstm)

# Get predictions from GRU model
predicted_closing_price_gru=gru_model.predict(x_test)
predicted_closing_price_gru=scaler.inverse_transform(predicted_closing_price_gru)

# Plot models predictions compared to valid data
plt.plot(actual_prices, color='black', label=f'Actual {company_id} price')
plt.plot(predicted_closing_price_lstm, color='green', label=f'Predicted LSTM {company_id} price')
plt.plot(predicted_closing_price_gru, color='yellow', label=f'Predicted GRU {company_id} price')
plt.title(f'{company_id} share price')
plt.xlabel('Time')
plt.ylabel(f'{company_id} share price')
plt.legend()
plt.savefig('logs\predict_graph.png')

# Predict day ahead stock price
data_real = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
data_real = np.array(data_real)
data_real = np.reshape(data_real, (data_real.shape[0], data_real.shape[1], 1))

# LSTM prediction
day_ahead_prediction_lstm = lstm_model.predict(data_real)
day_ahead_prediction_lstm = scaler.inverse_transform(day_ahead_prediction_lstm)

# GRU prediction
day_ahead_prediction_gru = gru_model.predict(data_real)
day_ahead_prediction_gru = scaler.inverse_transform(day_ahead_prediction_gru)

# Save prediction values to logs
with open('logs\stock_predictions.txt', 'w') as f:
    f.write(f'LSTM day ahead {company_id} stock price prediction: {day_ahead_prediction_lstm}\n')
    f.write(f'GRU day ahead {company_id} stock price prediction: {day_ahead_prediction_gru}\n')