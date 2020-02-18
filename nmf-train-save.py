# (setq python-shell-interpreter "~/python-environments/ml/bin/python")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline

from sklearn.datasets import fetch_20newsgroups


import csv
import numpy as np


def save_model(dictionary, feature_names, filename):
    """save two numpy arrays, one for the dictionary and one for the
    feature names. Can be loaded back in with load_model"""
    with open(filename, "wb") as f:
        np.savez(f, dictionary=dictionary, feature_names=feature_names)

def load_model(filename):
    """load model saved with save_model. Returns a tupe of dictionary, feature_names"""
    with open(filename, "rb") as f:
        arrs = np.load(f)
        return arrs["dictionary"], arrs["feature_names"]
    

def print_top_words(dictionary, feature_names, n_top_words):
    """print the top n_top_words from each topic in the dictionary matrix
    of dictionary. Feature names is a list of words with indices corresponding
    to the columns of the dictionary"""
    for topic_idx, topic in enumerate(dictionary):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def train_model(data, n_features, n_components):
    """train a model on data. Data is a list of strings to be trained on.
    They are converted into a bag of words representation and then
    transformed using a tfidf transformer with a maximum of n_features
    words in the model. This representation is then fed into the NMF
    algorithm which has n_components number of features. Returns the
    dictionary matrix (np array) and a list of words which represent the
    columns of the dictionary matrix"""
    model = Pipeline(steps=[("tfidf", TfidfVectorizer(max_df=0.95, min_df=2,
                                                      max_features=n_features, stop_words='english')),
                            ("nmf", NMF(n_components=n_components, random_state=1,
                                       alpha=.1, l1_ratio=.5))])
    model.fit(data)
    features = model["tfidf"].get_feature_names()
    dictionary = model["nmf"].components_
    return dictionary, features


def train_save_model(data, n_features, n_components, filename):
    """convenience function for training and saving a model. data should
    be a list of strings to train on, n_features is number of words in
    tfidf model, and n_components is number of nmf topics. Filename is
    where to save the model. Can be loaded later on with load_model"""
    dictionary, features = train_model(data, n_features, n_components)
    save_model(dictionary, features, filename)
    return dictionary, features



def load_shakes(filename):
    """load shakespear data from https://www.kaggle.com/kingburrito666/shakespeare-plays#alllines.txt"""
    with open(filename, "r") as f:
        data = [line.strip() for line in f]
        return data

def load_delta(filename, method="append"):
    """load delta data. Depending on method, either return question +
    answer separated with a space ('append'), question ('question'), or answer ('answer')"""
    with open(filename, "r", encoding = "utf8") as f:
        reader = csv.reader(f, delimiter='\t')
        data = []
        next(reader, None) # ignore headers
        for line in reader:
            if method == "append":
                data.append(line[1] + " " + line[2])
            elif method == "question":
                data.append(line[1])
            elif method == "answer":
                data.append(line[2])
            else:
                print("Error: method must be either 'append', 'question', or 'answer'")
                return None
        return data
        
def load_news():
    dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                                 remove=('headers', 'footers', 'quotes'))
    data_samples = dataset.data
    return data_samples


def main():
    # number of top words to use in tfidf representation of data
    n_features = 1000
    # number of topics in NMF
    n_components = 10


    # train model and save it
    print("Loading, training, and saving 20 news groups model")
    news_data = load_news()
    news_dict, news_feats = train_save_model(news_data, n_features, n_components, "C:/Users/wxwyl/Desktop/wylcode/nmf-train-save/news-nmf.npz")
    print_top_words(news_dict, news_feats, 10)
    
    # shakespeare data
    print("Loading, training, and saving Shakespeare model")
    shakes_data = load_shakes("C:/Users/wxwyl/Desktop/wylcode/nmf-train-save/data/alllines.txt")
    shakes_dict, shakes_feats = train_save_model(shakes_data, n_features, n_components, "C:/Users/wxwyl/Desktop/wylcode/nmf-train-save/shakes-nmf.npz")
    print_top_words(shakes_dict, shakes_feats, 10)

    # delta data
    print("Loading, training, and saving Delta model")
    delta_data = load_delta("C:/Users/wxwyl/Desktop/wylcode/nmf-train-save/data/delta-train.tsv")
    delta_dict, delta_feats = train_save_model(delta_data, n_features, n_components, "C:/Users/wxwyl/Desktop/wylcode/nmf-train-save/delta-nmf.npz")
    print_top_words(delta_dict, delta_feats, 10)


if __name__ == "__main__":
    main()
    #c = np.load("C:/Users/wxwyl/Desktop/wylcode/nmf-train-save/delta-nmf.npz")["dictionary"]
    #d = np.load("C:/Users/wxwyl/Desktop/wylcode/nmf-train-save/delta-nmf.npz")["feature_names"]
    #print(len(c[0]),len(c))
    #print(len(d))
    #print(c)
    #print(d)
