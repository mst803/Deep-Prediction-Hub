from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_review_length = 500
X_train = pad_sequences(X_train, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)

# Modelling a sample DNN
model = Sequential()
model.add(Embedding(input_dim=top_words, output_dim=24, input_length=max_review_length))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# opt=Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("Training Started.")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=20)

loss, acc = model.evaluate(X_test, y_test)
print("Training Finished.")

print(f'Test Accuracy: {round(acc * 100)}')

model.save(r'C:\Users\shahi\Desktop\My Projects\DeepPredictorHub\DP.keras')




