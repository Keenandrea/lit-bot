import string


# load text document in
# to memory return text
def load_doc(filename):
    # open file as a read only
    file = open(filename, 'r')
    # read entire text
    text = file.read()
    # close file
    file.close()
    return text


# takes loaded docu
# ment as an argume
# nt and returns an
# array of clean to
# kens
def clean_doc(doc):
    # replaces '--' with a space
    doc = doc.replace('--', ' ')
    # split into tokens
    # among white space
    tokens = doc.split()
    # remove punctuation from tokens to reduce vocab
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetical
    tokens = [word for word in tokens if word.isalpha()]
    # convert all tokens to lower case format
    tokens = [word.lower() for word in tokens]
    return tokens


# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# load cleaner version of the
# document in trial_clean.txt
in_filename = 'trial_clean.txt'
doc = load_doc(in_filename)
# print(doc[:200])

# run cleaner operation
tokens = clean_doc(doc)
# print(tokens[:200])
# print('Total Tokens: %d' % len(tokens))
# print('Unique Tokens: %d' % len(set(tokens)))

# organize into
# sequences of
# 50 input word
# s and 1 outpu
# t word
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i - length:i]
    # convert into a long
    # list of lines
    line = ' '.join(seq)
    # store the sequences
    sequences.append(line)
# print('Total Sequences: %d' % len(sequences))

# save the tokenized sequences to file
out_filename = 'trial_sequences.txt'
save_doc(sequences, out_filename)
