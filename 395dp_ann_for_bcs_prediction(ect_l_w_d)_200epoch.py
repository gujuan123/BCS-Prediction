import numpy
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#import pdb
for i in range(100):
    #def pdb_test():
    # load the dataset
    dataset = loadtxt('395 data (4input_ECT L W D).csv', delimiter=',')
    # split into input (X) and output (y) variables
    X = dataset[:,0:4]
    y = dataset[:,4]
    # reshape arrays into into rows and cols
    X = X.reshape(395, 4)
    y = y.reshape(395, 1)
    # separately scale the input and output variables
    scale_X = MinMaxScaler()
    X = scale_X.fit_transform(X)
    scale_y = MinMaxScaler()
    y = scale_y.fit_transform(y)
    #print(X.min(), X.max(), y.min(), y.max())
    # define the keras model
    model = Sequential()
    model.add(Dense(24, input_dim=4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    #pdb.set_trace()
    # split into 90% for train and 10% for test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=200,batch_size=10)


    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # make predictions for the input data
    predictions = model.predict(X)
    #X = (scale_X).inverse_transform(X)
    #y = (scale_y).inverse_transform(y)
    #excepts = (scale_y).inverse_transform(predictions)

    # rescale train data and test data
    predictions_train = model.predict(X_train)
    X_train = (scale_X).inverse_transform(X_train)
    y_train = (scale_y).inverse_transform(y_train)
    excepts_train = (scale_y).inverse_transform(predictions_train)
    predictions_test = model.predict(X_test)
    X_test = (scale_X).inverse_transform(X_test)
    y_test = (scale_y).inverse_transform(y_test)
    excepts_test = (scale_y).inverse_transform(predictions_test)
    X = (scale_X).inverse_transform(X)
    y = (scale_y).inverse_transform(y)
    excepts = (scale_y).inverse_transform(predictions)
    print('MSE of train data: %.3f' % mean_squared_error(y_train, excepts_train))
    print('MSE of test data: %.3f' % mean_squared_error(y_test, excepts_test))
    print('MSE: %.3f' % mean_squared_error(y, excepts))
    # summarize the train dataset and test dataset
    for i in range(264):
        print('train %s => %d (expected %d)' % (X_train[i].tolist(), excepts_train[i], y_train[i]))
    #print(f"X Train labels:\n{X_train}")
    #print(f"y Train labels:\n{y_train}")
    for i in range(131):
        print('test %s => %d (expected %d)' % (X_test[i].tolist(), excepts_test[i], y_test[i]))
    #print(f"X Test labels:\n{X_test}")
    #print(f"y Test labels:\n{y_test}")

    # report model error
    #print('MSE: %.3f' % mean_squared_error(y, excepts))
    print('MSE of train data: %.3f' % mean_squared_error(y_train, excepts_train))
    print('MSE of test data: %.3f' % mean_squared_error(y_test, excepts_test))
    print('MSE: %.3f' % mean_squared_error(y, excepts))
    #pdb_test()
