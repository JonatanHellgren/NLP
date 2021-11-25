from collections import Counter, defaultdict
from io import open
from matplotlib import pyplot as plt
import numpy as np



"""
Gettting the data
"""
def get_unigrams():

    books_freq = get_unigram('./a1_data/books.txt', 'ISO-8859-1')
    euparl_freq = get_unigram('./a1_data/europarl.txt', 'utf-8')
    wiki_freq = get_unigram('./a1_data/wikipedia.txt', 'utf-8')

    return [books_freq, euparl_freq, wiki_freq]


def get_unigram(file, encoding):
    freq = Counter()
    with open(file, encoding=encoding) as f: #, errors='ignore'
        for line in f:
            tokens = line.lower().split()
            for token in tokens:
                freq[token] += 1
    return freq


def get_bigrams():

    books_bigram = get_bigram('./a1_data/books.txt', 'ISO-8859-1')
    euparl_bigram = get_bigram('./a1_data/europarl.txt', 'utf-8')
    wiki_bigram = get_bigram('./a1_data/wikipedia.txt', 'utf-8')

    return [books_bigram, euparl_bigram, wiki_bigram]


def get_bigram(file, encoding):
    freqs = defaultdict(Counter)
    with open(file, encoding=encoding) as f:
        for line in f:
            tokens = line.lower().split()
            for t1, t2 in zip(tokens, tokens[1:]):
                freqs[t1][t2] += 1
    return freqs


"""
Wamup: computing word Frequencies
"""
def top_ten(freqs, names):
    for freq, name in zip(freqs, names):
        print("Frequencies for: ", name)
        for w, f in freq.most_common(10):
            print(w, f)
        print('\n')



def warmup():

    unigrams = get_unigrams()
    top_ten(unigrams, ['Books', 'Euparl', 'Wiki'])

    bigrams = get_bigrams()
    bigrams_red = [bigrams[0]['red'], bigrams[1]['red'], bigrams[2]['red']]
    top_ten(bigrams_red, ['Books', 'Euparl', 'Wiki'])


"""
Investigation the word frequency distribution
"""
def freq_distribution_plot():
    unigrams = get_unigrams()
    summed_frequencies = (unigrams[0] + unigrams[1] + unigrams[2])
    top = 100
    most_common = summed_frequencies.most_common(top)

    frequencies = []
    tokens = []

    for it in range(top):
        frequencies.append(most_common[it][1])
        tokens.append(most_common[it][0])

    plt.scatter(range(1,top+1), frequencies)
    plt.loglog()
    plt.show()



"""        
Side show: preprocessing text for machine learning
def preprocessing(dataset):
    with open(dataset) as f:
        voc = Vocab(max_voc_size=1000, batch_size=8)
        voc.build_vocab(f)

    with open(dataset) as f:
        for b in voc.batches(f):
            print(b)

"""

"""
Trying ou an NLP toolkit
"""
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')


def manual_preprocessing():
    example = 'Apple bought two companies this year and no one knew, Mark Gurman at 9to5Mac reports.'
    result = nlp(example)
    
    html = displacy.render(result, style='dep', page=True)

    with open('result.html', 'w') as f:
        f.write(html)

    print("\nTokens:")
    for token in result:
        if token.dep_ == "dobj":
            print(token)
            print(token.head.lemma_)

    # print("\nEntities:")
    # for entity in result.ents:
        # print(entity)


"""
Which are the most frequent nouns in the book review corpus
"""
def most_freq_noun(file):

    freq = Counter()                                    # counter to store frequencies in

    nlp = spacy.load('en_core_web_sm')

    with open(file) as f:
        for line in f:
            tokens = nlp.pipe(line, n_process=-1)
            for token in tokens:
                print(token)






    # with open(file, encoding='ISO-8859-1') as f:        # opening the file
        # for doc in nlp.pipe(f, disable=["tokenizer", "parser", "ner", "lemmatizer", "textcat"], n_process=-1):
        # for doc in nlp.pipe(f, pipeline=["tagger"], n_process=-1):
            # Do something with the doc here
            # for token in doc:
                # if token.pos_ == "NOUN":
                    # freq[token.text] += 1

    return freq                                         # returns the resulting counter
        
def most_mentioned_country(file):

    freq = Counter()

    with open(file, encoding='utf-8') as f:
        for line in f:                                  # for each line
            result = nlp(line)                          # convert to en_core_web_sm
            for token in result.ents:                        # for each token
                if token.label_ == "GPE":
                    freq[token.text] += 1                    # add one in the frequency counter

    return freq

def most_freq_drink(file):

    freq = Counter()

    with open(file, encoding='utf-8') as f:
        for line in f:                                  # for each line
            if "drink" in line or "drinking" in line:
                result = nlp(line)                          # convert to en_core_web_sm
                for token in result:                        # for each token
                    if token.dep_ == "dobj" and token.head.lemma_ == "drink":
                        freq[token.text] += 1                    # add one in the frequency counter
    return freq



def read_file(file):
    with open(file) as f:
        for line in f:
            result = nlp.pipe(line, n_process=-1)

    











    
