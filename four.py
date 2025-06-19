import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras

def prob(num1, num2):
    mp = 'four.keras'
    x = np.linspace(-10,10, 100)
    z = np.linspace(-10,10, 100)
    X,Z = np.meshgrid(x,z)
    X = X.flatten()
    Z = Z.flatten()
    Y = np.exp(-0.1 * (X**2 + Z**2))
    # yn = Y.min()
    # yx = Y.max()
    # xn = X.min()
    # xx = X.max()
    # zn = Z.min()
    # zx = Z.max()
    # Y_norm = (Y-yn)/(yx-yn)
    # X = (X-xn)/(xx-xn)
    # Z = (Z-zn)/(zx-zn)
    inp = np.column_stack((X,Z))
    if (os.path.exists(mp)):
        model = keras.models.load_model(mp)
    else:
        model = keras.Sequential([keras.layers.Dense(32, input_shape=(2,), activation = 'relu'),
                                  keras.layers.Dense(16, activation='relu'),
                                  keras.layers.Dense(units=1, activation='sigmoid')])
        model.compile(optimizer = 'Adam', loss='mean_squared_error')
        hist = model.fit(inp, Y, epochs=100)
        lv = hist.history['loss']
        pred = model.predict(inp)
        model.save('four.keras')
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
    return model.predict(np.array([[num1,num2]]))[0][0]

print(prob(0,0))
print(prob(5,5))
print(prob(10,10))