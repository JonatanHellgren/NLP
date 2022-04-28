import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import  Counter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def load_data(file='../data/a3_data/wsd_train.txt'):
    """
    Loading the data
    """
    X = []
    y = []
    w = []
    p = []
    with open(file) as f:
        for line in f:
            label, word, pos, text = line.split(maxsplit=3)
            X.append(text)
            y.append(label)
            w.append(word)
            p.append(pos)
    return X, y, w, p


def encode_labels(labels, return_encoder = False):
    """
    Encoding the labels as integers
    """

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    labels_encoded = label_encoder.transform(labels)

    if not return_encoder:
        return labels_encoded

    return labels_encoded, label_encoder

def center(X, p, pad_sz):
    """
    This function centers each target word in the data.
    It adds padding tokens if necessary.
    """
    X_centered = []
    for ind, pind in enumerate(p):
        pind = int(pind)
        start_ind = max(pind-pad_sz, 0)
        end_ind = min(pind+pad_sz+1, len(X[ind].split()))
        X_centered.append(X[ind].split()[start_ind:end_ind])

        for _ in range(max(pad_sz-pind, 0)):
            X_centered[ind].insert(0, '<empty>')

        for _ in range(max(pind+pad_sz+1-len(X[ind].split()), 0)):
            X_centered[ind].append('<empty>')

    return X_centered


def get_dictionary(X, vocab_size):
    """
    This function constructs a dictionary by counting the frequencies of each word and 
    giving each word an encoding that corresponds to how common it is. Words out of
    vocabulary will be considered as zeros, the same is for the padding tokens.
    """
    word_freq = Counter()
    for document in X:
        for word in document:
            if word != '<empty>':
                word_freq[word] += 1

    dictionary = Counter()
    for ind, freq in enumerate(word_freq.most_common(vocab_size)):
        dictionary[freq[0]] += ind + 1

    return dictionary


def onecold_encode(X, dictionary, pad_sz):
    """
    This function takes a dictionary and som data and encodes each token as the ingeger value
    in the dictionary.
    """
    X_onecold = np.zeros((len(X), pad_sz*2 + 1), dtype=int)
    for ind, doc in enumerate(X):
        for jnd, token in enumerate(doc):
            X_onecold[ind, jnd] = dictionary[token]

    return X_onecold



def encode_data(vocab_size, X, p, X_test, p_test, pad_sz=20):
    """
    This function encodes the data we use as input for the model.
    """

    # First center and pad
    X_centered = center(X, p, pad_sz)
    X_test_centered = center(X_test, p_test, pad_sz)

    # Then create a dictionary
    dictionary = get_dictionary(X_centered, vocab_size)

    # After that we will onecold encode them acording to our dictionary
    X_onecold = onecold_encode(X_centered, dictionary, pad_sz)
    X_test_onecold = onecold_encode(X_test_centered, dictionary, pad_sz)

    return X_onecold, X_test_onecold


def get_dataloaders(vocab_size, padding_size, val_size, batch_size, random_state):
    """
    This function creates the dataloaders used for training the model.
    """
    X, y, w, p = load_data()
    X_test, y_test, w_test, p_test = load_data('../data/a3_data/wsd_test_blind.txt')

    X_encoded, X_test_encoded = encode_data(vocab_size, X, p, X_test, p_test, pad_sz=padding_size)


    # Encoding y
    y_encoded, label_encoder = encode_labels(y, return_encoder=True)

    w_encoded = encode_labels(w)
    w_test_encoded = encode_labels(w_test)

    # Splitting the data into train and test
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X_encoded,
            y_encoded, w_encoded, test_size=val_size, random_state=random_state)

    # And constructing dataloaders
    train_dataset = list(zip(X_train, y_train, w_train))
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = list(zip(X_val, y_val, w_val))
    val_dataloader = DataLoader(val_dataset,  batch_size, shuffle=False)

    test_dataset = list(zip(X_test_encoded, y_test, w_test_encoded))
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    # We return the dataloader as well as the label encoder since we need that to translate the predictions of our model
    return train_dataloader, val_dataloader, test_dataloader, label_encoder


