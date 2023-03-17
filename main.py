import glob
import os

import keras.models
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.api as sm

from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.models import Sequential, load_model
from keras import layers
from keras.layers import *
from sklearn.model_selection import train_test_split
from tslearn.clustering import KShape


def preprocessing(file):
    sb.set(rc={'figure.figsize': (10, 4)})

    df = pd.read_csv(file)
    df.plot()
    plt.title('Raw data')

    # concatenating date and time columns
    df['date'] = df['date'] + ' ' + df['time']

    # dropping time column since it's no longer needed
    df = df.drop(columns='time')

    # setting date as index
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)

    # checking type of date column
    print(df.index.dtype)

    # replacing non-valid values with NaN so that we can fill the spaces up
    df = df.replace(-1, np.nan)
    print(df.tail(15))

    # applying forward and backward fill
    data = df.ffill().bfill()
    print(data.tail(15))

    plt.figure()
    plt.plot(data)
    plt.title('30 sec speed')
    plt.xlabel('Date')
    plt.ylabel('Speed')

    plt.figure()
    decomposition = sm.tsa.seasonal_decompose(data, period=42 * 24)
    decomposition.seasonal.plot()
    plt.title('Seasonality')

    plt.figure()
    decomposition.trend.plot()
    plt.title('Trend')

    plt.figure()
    plt.plot(data)
    data = pd.Series.ewm(data['6908'], alpha=0.03, adjust=False).mean()
    plt.plot(data)
    plt.title('Exponential Smoothing Filter - Noise Reduction')

    plt.show()
    return data


# converting dataframe to numpy array
# window_size gives us how far we want to look back
# [[[1], [2], [3]]] [4]
# X:[[[2], [3], [4]]] y:[5]
def df_to_array(df, window_size):
    df_as_np = df.to_numpy()
    X = []
    y = []

    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size]
        y.append(label)

    return np.array(X), np.array(y)


def learning_curve(model):
    plt.plot(model.history.history['loss'][:15], label='training loss')
    plt.plot(model.history.history['val_loss'][:15], label='validation loss')
    plt.legend()
    plt.show()


def training_predictions_plot(Xtrain, ytrain, model):
    train_p = model.predict(Xtrain).flatten()
    train_r = pd.DataFrame(data={'Train Preds': train_p, 'Actuals': ytrain})

    plt.plot(train_p, linewidth=0.4, label='Predictions')
    plt.plot(ytrain, linewidth=0.3, label='Actuals')
    plt.title('Train Prediction')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_r['Train Preds'][:100], linewidth=0.4, label='Predictions')
    plt.plot(train_r['Actuals'][:100], linewidth=0.3, label='Actuals')
    plt.title('Train Prediction')
    plt.legend()
    plt.show()


def validation_predictions_plot(Xval, yval, model):
    val_p = model.predict(Xval).flatten()
    val_r = pd.DataFrame(data={'Validation Preds': val_p, 'Actuals': yval})

    plt.plot(val_p, linewidth=0.4, label='Predictions')
    plt.plot(yval, linewidth=0.3, label='Actuals')
    plt.title('Validation Prediction')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(val_r['Validation Preds'][:100], linewidth=0.4, label='Predictions')
    plt.plot(val_r['Actuals'][:100], linewidth=0.3, label='Actuals')
    plt.title('Validation Prediction')
    plt.legend()
    plt.show()


def test_prediction_plot(Xtest, ytest, model):
    test_p = model.predict(Xtest).flatten()
    test_r = pd.DataFrame(data={'Test Preds': test_p, 'Actuals': ytest})

    plt.plot(test_p, linewidth=0.4, label='Predictions')
    plt.plot(ytest, linewidth=0.3, label='Actuals')
    plt.title('Test Prediction')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(test_r['Test Preds'][:100], linewidth=0.4, label='Predictions')
    plt.plot(test_r['Actuals'][:100], linewidth=0.3, label='Actuals')
    plt.title('Test Prediction')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    data1 = preprocessing('30sec_speeds-20220310.csv')
    data2 = preprocessing('30sec_speeds-20220311.csv')
    data3 = preprocessing('30sec_speeds-20220312.csv')
    data4 = preprocessing('30sec_speeds-20220313.csv')
    data5 = preprocessing('30sec_speeds-20220314.csv')
    data6 = preprocessing('30sec_speeds-20220315.csv')

    data = pd.concat([data1, data2, data3, data4, data5, data6], ignore_index=False)

    window_size = 100
    X, y = df_to_array(data, window_size)
    print(X.shape, y.shape)

    # setting train, validation and test sets
    X_train, y_train = X[:12000], y[:12000]
    X_val, y_val = X[12000:14500], y[12000:14500]
    X_test, y_test = X[14500:], y[14500:]

    # no. of columns i have
    n_features = 1

    model1 = Sequential()
    model1.add(LSTM(100, input_shape=(window_size, n_features)))
    model1.add(Dense(8, 'relu'))
    model1.add(Dense(1))

    print(model1.summary())

    # saves the model with the best performance
    cp = ModelCheckpoint('model/', save_best_only=True)
    model1.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[RootMeanSquaredError()])

    model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[cp])

    learning_curve(model1)

    # loads the saved model
    model1 = load_model('model/')

    training_predictions_plot(X_train, y_train, model1)
    validation_predictions_plot(X_val, y_val, model1)
    test_prediction_plot(X_test, y_test, model1)

    # GRU
    model2 = Sequential()
    model2.add(GRU(100, input_shape=(window_size, n_features)))
    model2.add(layers.Dropout(rate=0.2))
    model2.add(Dense(1))

    print(model2.summary())

    # saves the model with the best performance
    cp2 = ModelCheckpoint('model2/', save_best_only=True)
    model2.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[RootMeanSquaredError()])

    model2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[cp2])

    learning_curve(model2)

    # loads the saved model
    model2 = load_model('model2/')

    training_predictions_plot(X_train, y_train, model2)
    validation_predictions_plot(X_val, y_val, model2)
    test_prediction_plot(X_test, y_test, model2)

    # GRU2
    modelgru = Sequential()
    modelgru.add(GRU(100, input_shape=(window_size, n_features), recurrent_dropout=0.2))
    modelgru.add(layers.Dropout(rate=0.2))
    modelgru.add(Dense(1))

    print(modelgru.summary())

    # saves the model with the best performance
    cp2 = ModelCheckpoint('modelgru/', save_best_only=True)
    modelgru.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[RootMeanSquaredError()])

    modelgru.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[cp2])

    learning_curve(modelgru)

    # loads the saved model
    modelgru = load_model('modelgru/')

    training_predictions_plot(X_train, y_train, modelgru)
    validation_predictions_plot(X_val, y_val, modelgru)
    test_prediction_plot(X_test, y_test, modelgru)


