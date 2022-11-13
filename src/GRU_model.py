from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense


class GRU_adam_model():
    def __init__(self, epochs):
        self.epochs = epochs

    def getModel(self, x_train_data, y_train_data):
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(GRU(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train_data, y_train_data, epochs=self.epochs, batch_size=50, verbose=1)
        return model