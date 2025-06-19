import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras

def prob(num1, num2):
    mp = 'three.keras'
    x = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, 2*np.pi, 100)
    X,Z = np.meshgrid(x,z)
    X = X.flatten()
    Z = Z.flatten()
    Y = 2*np.sin(X) + 3*np.cos(Z)
    inp = np.column_stack((X,Z))
    if (os.path.exists(mp)):
        model = keras.models.load_model(mp)
    else:
        model = keras.Sequential([keras.layers.Dense(32, input_shape=(2,), activation = 'tanh'),
                                  keras.layers.Dense(16, activation='tanh'),
                                  keras.layers.Dense(units=1)])
        model.compile(optimizer = 'Adam', loss='mean_squared_error')
        hist = model.fit(inp, Y, epochs=100)
        lv = hist.history['loss']
        pred = model.predict(inp)
        model.save('three.keras')
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
    return model.predict(np.array([[num1, num2]]))[0][0]

print(prob(np.pi/2, 0))
print(prob(np.pi, np.pi))
print(prob(3*np.pi/2, np.pi/2))