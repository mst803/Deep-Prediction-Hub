from tensorflow.keras.datasets import imdb
from BackPropogation import BackPropogation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import pickle

top_words = 5000
(X_train, y_train), (X_test,y_test) = imdb.load_data(num_words=top_words)

max_review_length = 500

X_train = pad_sequences(X_train, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)

backprop = BackPropogation(epochs=100,learning_rate=0.01,activation_function='sigmoid')
backprop.fit(X_train, y_train)
pred = backprop.predict(X_test)
print(f"Accuracy : {accuracy_score(pred, y_test)}")

with open(r'C:\Users\shahi\Desktop\My Projects\DeepPredictorHub\BP.pkl','wb') as file:
    pickle.dump(backprop, file)