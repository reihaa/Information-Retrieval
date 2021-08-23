import nltk
import matplotlib.pyplot as plt
import numpy as np

nltk.download('movie_reviews')
nltk.download('stopwords')

from nltk.corpus import movie_reviews, stopwords
import string

words_bag = list(map(str.lower, movie_reviews.words()))

english_stopwords = stopwords.words('english')
english_punctuations = list(string.punctuation)
remove_words = english_stopwords + english_punctuations

words_bag = [word for word in words_bag if word not in remove_words]

terms = dict()
terms_len = 0
X_h = []  #term number
Y_h = []  # word number
for i, token in enumerate(words_bag):
    if token in terms:
        terms[token] += 1
    else:
        Y_h.append(i+1)
        terms[token] = 1
        terms_len += 1
        X_h.append(terms_len)


def heap():
    x, y = np.log10(X_h), np.log10(Y_h)
    plt.plot(x, y, label="heap")
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, label="estimated line")
    print("estimated line equation for heap : {}x + {}".format(m, b))
    plt.xlabel('log10 T')
    plt.ylabel('log10 M')
    plt.legend()
    plt.show()


def zipf():
    Y = list(reversed(sorted(terms.values())))  #frequecy
    X = np.arange(1, terms_len+1)  #rank
    x, y = np.log10(X), np.log10(Y)
    plt.plot(x, y, label="zipf")
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, label="estimated ")
    print("estimated line equation for zipf: {}x + {}".format(m, b))
    plt.xlabel('log10 rank')
    plt.ylabel('log10 cf')
    plt.legend()
    plt.show()

zipf()
# heap()
