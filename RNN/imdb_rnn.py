import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf


from numpy import argmax
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import pickle

top_words = 5000
(X_train, y_train), (X_test,y_test) = imdb.load_data(num_words=top_words)

max_review_length = 500
X_train = pad_sequences(X_train, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)

model=tf.keras.models.Sequential([
   tf.keras.layers.Embedding(input_dim=top_words,output_dim= 24, input_length=max_review_length),
   tf.keras.layers.SimpleRNN(24, return_sequences=False),
   tf.keras.layers.Dense(64, activation='relu'),
   tf.keras.layers.Dense(32, activation='relu'),
   tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("----------------------  -------------------------\n")

# summarize the model
print(model.summary())

print("----------------------  -------------------------\n")

early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', patience=10)

print("----------------------  Training -------------------------\n")

# fit the model
model.fit(x=X_train,
         y=y_train,
         epochs=100,
         validation_data=(X_test, y_test),
         callbacks=[early_stop]
         )
print("----------------------  -------------------------\n")


def acc_report(y_true, y_pred):
   acc_sc = accuracy_score(y_true, y_pred)
   print(f"Accuracy : {str(round(acc_sc,2)*100)}")
   return acc_sc


preds = (model.predict(X_test) > 0.5).astype("int32")
print(acc_report(y_test, preds))

model.save(r'C:\Users\shahi\Desktop\My Projects\DeepPredictorHub\RN.keras')