import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.svm import SVC

stop_words = set(stopwords.words('english'))

#read data and put it into two columns : ['target', 'text']
def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset


# make data ready for process
def preprocess(tweet):
    tweet.lower()
    # lorwercasing letters so the same words wont count as different words

    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # delete links becuase they don't add meaning

    tweet = re.sub(r'\@\w+|\#\w+', '', tweet)
    # delete hashtags and nametags(usually it's not a right thing to do but here only two hashtags were used)

    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # delete punctuations

    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    # delete stopwords

    result = " ".join(filtered_words)
    return result


# make a vector from train data
def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector


dataset = load_dataset("StrictOMD.csv", ['target', 'text'])
dataset.text = dataset['text'].apply(preprocess)
tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())

# second column including tweet texts
X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())

# first column including 0 and 4
y = np.array(dataset.iloc[:, 0]).ravel()

# shuffle data then divide it into 5 and process it 5 times, each time taking one as test data and train with the
# other 4
cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
SVC_model = SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(SVC_model, X, y, cv=cv)

# print five outcome, the minimum is the fitness
print("Support vector machine:", scores, sep='\t')

#outcome:
#Support vector machine: [0.78165939 0.81659389 0.82532751 0.78165939 0.83406114]
