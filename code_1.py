import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense,Embedding,SimpleRNN
from keras.preprocessing.text import text_to_word_sequence,Tokenizer
from keras .utils import pad_sequences
from keras.models import Sequential
docs = ["pakistan zindabad",
        "i love pakistan",
        "pakistan won the game",
        "pakistan beat the rivalry",
        "england lies every morning"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)
print(len(tokenizer.word_index))
sequences = tokenizer.texts_to_sequences(docs)
sequences = pad_sequences(sequences,padding="post")
model = Sequential()
model.add(Embedding(32,output_dim=2,input_length=4))
print(model.summary())
model.compile(optimizer="adam",metrics=["accuracy"])
predict = model.predict(sequences)
print(predict)
