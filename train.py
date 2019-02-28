from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


# load text document in
# to memory; return text
def load_doc(filename):
    # open file as a read only
    file = open(filename, 'r')
    # read entire text
    text = file.read()
    # close file
    file.close()
    return text


# load the tokenized sequences file
in_filename = 'trial_sequences.txt'
doc = load_doc(in_filename)
# split data into seper
# ate training sequence
# s by splitting at new
# lines
lines = doc.split('\n')

# the word embedding la
# yer expects input seq
# uences to be comprise
# d of integers. so we
# map each word in our
# vocabulary to a uniqu
# e integer and encode
# our input sequences.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
# determine the size of our text vocabulary
vocab_size = len(tokenizer.word_index) + 1

# separate into input(X) and
# output (Y) elements by arr
# ay slicing
sequences = array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
# one hot encode the output word. convert it
# from an integer to a vector of 0 values, o
# ne for each word in the vocabulary, with a
# 1 to indicate the specific word at the ind
# ex of the words integer value. by this the
# model learns to predict the probability di
# stribution for the next word and the groun
# nd truth from which to learn from is 0 for
# all words except the actual word that come
# s next. to_categorical() is provided by Ke
# ras to one hot encode the output words for
# each input-output sequence pair.
y = to_categorical(y, num_classes=vocab_size)
# specify to the embedd
# ing layer how long in
# put sequences are. on
# e generic way of doin
# g this is to use the
# second dimension (num
# ber of columns) of th
# e input data's shape.
seq_length = X.shape[1]

# define and fit our
# language model on
# the training data.
model = Sequential()
# common values are 50, 100, and 300. consider smaller or lar
# ger values for testing.
model.add(Embedding(vocab_size, 50, input_length=seq_length))
# start with two LSTM hidden layers with 100
# memory cells each. more memory cells and a
# deeper network may achieve better results.
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
# a dense fully connected layer with 100
# neurons connects to the LSTM hidden la
# yers to interpret the features extract
# ed from the sequence.
model.add(Dense(100, activation='relu'))
# the output layer (binary) predicts the next word
# as a single vector the size of the vocabulary wi
# th a probability for each word in the vocabulary
# and softmax activation is used to ensure the out
# puts have the characteristics of normalized prob
# abilities.
model.add(Dense(vocab_size, activation='softmax'))
# summary printed as a
# sanity check to ensu
# re we have construct
# ed what we intended.
print(model.summary())
# model is compiled specifying the categorical cross entropy loss needed to fit the mo
# del because the model is learning multi-class classification. adam is used to mini-b
# atch gradient descent.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model is fit on the data for training 100
# epochs with a batch size of 128 for speed
model.fit(X, y, batch_size=128, epochs=100)

# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
