import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras

def prob(num1, num2):
    mp = 'one.keras'
    x = np.arange(0,101)
    z = np.arange(0,101)
    X,Z = np.meshgrid(x,z)
    X = X.flatten()
    Z = Z.flatten()
    inp = np.column_stack((X,Z))
    Y = 7*X + 3*Z-5
    yn = Y.min()
    yx = Y.max()
    Y_norm = (Y-yn)/(yx-yn)
    if (os.path.exists(mp)):
        model = keras.models.load_model(mp)
    else:
        model = keras.Sequential([keras.layers.Dense(32, input_shape=(2,), activation = 'relu'),
                                  keras.layers.Dense(16, activation='relu'),
                                  keras.layers.Dense(units=1)])
        model.compile(optimizer = 'Adam', loss='mean_squared_error')
        hist = model.fit(inp, Y_norm, epochs=100)
        lv = hist.history['loss']
        pred = model.predict(inp)
        model.save('one.keras')
        plt.figure()
        plt.plot(lv)
        plt.show()
        plt.figure()
        plt.scatter(Y, pred, label='actual vs predictions', color='red')
        plt.xlabel('actual')
        plt.ylabel('predicted')
        plt.legend()
        plt.grid(True)
        plt.show()
    return model.predict(np.array([[num1,num2]]))[0][0]*(yx-yn)+yn

print(prob(10,20))
print(prob(50,50))
print(prob(100,0))