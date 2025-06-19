import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras

def prob(num1, num2):
    mp = 'five.keras'
    x = np.linspace(0,50, 100)
    z = np.linspace(0,50,100)
    X,Z = np.meshgrid(x,z)
    X = X.flatten()
    Z = Z.flatten()
    Y = X*Z + 20
    yn = Y.min()
    yx = Y.max()
    xn = X.min()
    xx = X.max()
    zn = Z.min()
    zx = Z.max()
    Y_norm = (Y-yn)/(yx-yn)
    X = (X-xn)/(xx-xn)
    Z = (Z-zn)/(zx-zn)
    inp = np.column_stack((X,Z))
    if (os.path.exists(mp)):
        model = keras.models.load_model(mp)
    else:
        model = keras.Sequential([keras.layers.Dense(32, input_shape=(2,), activation = 'tanh'),
                                  keras.layers.Dense(16, activation='tanh'),
                                  keras.layers.Dense(units=1, activation='relu')])
        model.compile(optimizer = 'Adam', loss='mean_squared_error')
        hist = model.fit(inp, Y_norm, epochs=100)
        lv = hist.history['loss']
        pred = model.predict(inp)
        model.save('five.keras')
        plt.figure()
        plt.plot(lv)
        plt.show()
        plt.figure()
        plt.scatter(Y_norm, pred, label='actual vs predictions', color='red')
        plt.xlabel('actual')
        plt.ylabel('predicted')
        plt.legend()
        plt.grid(True)
        plt.show()
    return model.predict(np.array([[(num1-xn)/(xx-xn),(num2-zn)/(zx-zn)]]))[0][0]*(yx-yn)+yn

print(prob(10,2))
print(prob(0,30))
print(prob(5,5))