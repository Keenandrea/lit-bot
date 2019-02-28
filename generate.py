from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# load doc into memory
def load_doc(filename):
    # open file as read only
    file = open(filename, 'r')
    # read all of text
    text = file.read()
    # close file
    file.close()
    return text


# generate a sequence from a language model taking in the model, th
# e tokenizer, input sequence length, the seed text, and the number
# of words to generate. it returns a sequence of words generated by
# the model.
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length to keep the sequences from gett
        # ing too long.
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict the next word directly by calling func
        # tion that will return the index of the word wi
        # th the highest probability.
        yhat = model.predict_classes(encoded, verbose=0)

        out_word = ''
        # look up the index in the Tokenizers mapping to
        # get the associated word. then append this word
        # to the seed text and repeat the process.
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


# load cleaned text sequences
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
# specify the expected length of input
# by calculating the length of one lin
# e of the loaded data and subtracting
# 1 for the expected output word that
# is also on the same line.
seq_length = len(lines[0].split()) - 1

# load the model with Keras
model = load_model('model.h5')

# load the tokenizer from file using Pickle
tokenizer = load(open('tokenizer.pkl', 'rb'))

# select a random line of text from the
# input text and print it so that we ha
# ve some idea of what was used. seed t
# ext must be encoded to integers.
seed_text = lines[randint(0, len(lines))]
print(seed_text + '\n')

# generate a sequence of new words given some seed text.
generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
print(generated)
