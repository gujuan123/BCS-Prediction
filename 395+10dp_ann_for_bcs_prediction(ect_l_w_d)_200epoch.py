
import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt

# Loop multiple times if you're testing for stability
for i in range(70):
    # === Load and prepare training data ===
    dataset1 = loadtxt('395 data (4input_ECT L W D).csv', delimiter=',')
    X_train = dataset1[:, 0:4]
    y_train = dataset1[:, 4].reshape(-1, 1)

    # === Load and prepare test data ===
    dataset2 = loadtxt('Physical tested data(10 dimensions).csv', delimiter=',')
    X_test = dataset2[:, 0:4]
    y_test = dataset2[:, 4].reshape(-1, 1)

    # === Fit scalers on training data only ===
    scale_X = MinMaxScaler()
    scale_y = MinMaxScaler()
    X_train_scaled = scale_X.fit_transform(X_train)
    y_train_scaled = scale_y.fit_transform(y_train)

    # === Transform test data using same scalers ===
    X_test_scaled = scale_X.transform(X_test)
    y_test_scaled = scale_y.transform(y_test)

    # === Define and compile the model ===
    model = Sequential()
    model.add(Dense(24, input_dim=4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # === Train the model ===
    history = model.fit(X_train_scaled, y_train_scaled, validation_data=(X_test_scaled, y_test_scaled), epochs=200, batch_size=10)

    # === Plot training history ===
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # === Predictions and inverse scaling ===
    predictions_train_scaled = model.predict(X_train_scaled)
    predictions_test_scaled = model.predict(X_test_scaled)

    X_train_original = scale_X.inverse_transform(X_train_scaled)
    y_train_original = scale_y.inverse_transform(y_train_scaled)
    predictions_train_original = scale_y.inverse_transform(predictions_train_scaled)

    X_test_original = scale_X.inverse_transform(X_test_scaled)
    y_test_original = scale_y.inverse_transform(y_test_scaled)
    predictions_test_original = scale_y.inverse_transform(predictions_test_scaled)

    # === Calculate and print errors ===
    sum_y_train_error = 0
    for i in range(len(X_train_original)):
        print('Train %s => %.2f (expected %.2f)' % (X_train_original[i].tolist(), predictions_train_original[i], y_train_original[i]))
        sum_y_train_error += abs(predictions_train_original[i] - y_train_original[i]) / predictions_train_original[i]
    average_y_train_error = sum_y_train_error / len(X_train_original)
    print("Average train error:", average_y_train_error)

    sum_y_test_error = 0
    for i in range(len(X_test_original)):
        print('Test %s => %.2f (expected %.2f)' % (X_test_original[i].tolist(), predictions_test_original[i], y_test_original[i]))
        sum_y_test_error += abs(predictions_test_original[i] - y_test_original[i]) / predictions_test_original[i]
    average_y_test_error = sum_y_test_error / len(X_test_original)
    print("Average test error:", average_y_test_error)

    print('MSE of train data: %.3f' % mean_squared_error(y_train_original, predictions_train_original))
    print('MSE of test data: %.3f' % mean_squared_error(y_test_original, predictions_test_original))
