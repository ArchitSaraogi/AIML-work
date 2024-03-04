import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.activations import linear, relu, sigmoid



# Read the CSV file into a DataFrame
df = pd.read_csv('NIFTY50_all.csv')

# Specify the headers of the columns you want to convert
column_headers = ['Open', 'High', 'Low','Close']

# Convert the specified columns into NumPy arrays
arrays = [0,0,0]
for i in range(len(column_headers)-1):
    arrays[i] = np.array(df[column_headers[i]][-15:],dtype=np.float64)

arrays = np.array(arrays)
arrays = arrays.transpose()
yt = np.array(df[column_headers[3]][-15:],dtype=np.float64)
yt = yt.transpose()
print(yt)
norm_l = tf.keras.layers.Normalization(axis=-1)

norm_l.adapt(arrays)

model = Sequential(
    [
        tf.keras.Input(shape=(3,)),
        Dense(1, activation='ReLU', name = 'layer1')
     ]
)
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    arrays,yt,           
    epochs=10,
)
l=[[22290.00,22297.50,22186.10]]
l=np.array(l)

predictions = model.predict(l)
print(predictions)



    
        









