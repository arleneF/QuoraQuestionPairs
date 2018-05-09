# Reference: word embedding:
# http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/
# https://yq.aliyun.com/articles/221681
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import KFold
# from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.noise import GaussianNoise
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import xgboost as xgb

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=""

np.random.seed(1024) #makes the random numbers predictable
WNL = WordNetLemmatizer()
#even ImageNet is build on WordNet
STOP_WORDS = set(stopwords.words('english'))
MAX_SEQUENCE_LENGTH = 30
MIN_WORD_OCCURRENCE = 100
REPLACE_WORD = "memento"
EMBEDDING_DIM = 300
NUM_FOLDS = 5
BATCH_SIZE = 512 #1025
EMBEDDING_FILE = "data/glove.840B.300d.txt"


def cutter(word):
    if len(word) < 4:
        return word
    return WNL.lemmatize(WNL.lemmatize(word, "n"), "v")  #first change to noun then change to verb


def preprocess(string):
    string = string.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't"," not").replace("what's", "what is").replace("it's", "it is").replace("wasn't","was not") \
        .replace("hasn't","has not").replace("wouldn't","would not").replace("isn't", "is not") \
        .replace("shouldn't","should not").replace("weren't","were not").replace("couldn't","could not") \
        .replace("didn't","did not").replace("aren't","are not").replace("needn't","need not") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are").replace("mighn't","might not") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("bodylanguage","body language").replace("englisn","english") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")\
		.replace("e-mai", " email ").replace("'d","would")
    # string = re.sub('[“”\(…\)\!\^\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)
    string = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)
    string = re.sub(r"([0-9]+)000000", r"\1m", string)
    string = re.sub(r"([0-9]+)000", r"\1k", string)
	 #there are some typo, need to be corrected
    string = re.sub(r" e g ", " eg ", string)
    string = re.sub(r" b g ", " bg ", string)
    string = re.sub(r"\0s", "0", string)
    string = re.sub(r" 9 11 ", "911", string)
    string = re.sub(r"quikly", "quickly", string)
    string = re.sub(r"imrovement", "improvement", string)
    string = re.sub(r"intially", "initially", string)
    string = re.sub(r" dms ", "direct messages ", string)
    string = re.sub(r"demonitization", "demonetization", string)
    string = re.sub(r"kms", " kilometers ", string)
    string = re.sub(r" cs ", " computer science ", string)
    string = re.sub(r" upvotes ", " up votes ", string)
    string = re.sub(r"\0rs ", " rs ", string)
    string = re.sub(r"calender", "calendar", string)
    string = re.sub(r"programing", "programming", string)
    string = re.sub(r"bestfriend", "best friend", string)
    string = re.sub(r"iii", "3", string) #III
    string = re.sub(r" j k ", " jk ", string)
    string = ' '.join([cutter(w) for w in string.split()])
    return string


def get_embedding():
    embeddings_index = {}
    f = open(EMBEDDING_FILE)
    for line in f:
        values = line.split()
        word = values[0]
        if len(values) == EMBEDDING_DIM + 1 and word in top_words:
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def is_numeric(s):
    return any(i.isdigit() for i in s)



# s = "-";
# seq = ("a", "b", "c"); # This is sequence of strings.
# print s.join( seq )
# ==> a-b-c
def prepare(q):
    new_q = [] #the list only contains words in top_words and memento(if element is not in STOP_WORDS)
    surplus_q = []
    numbers_q = []
    new_memento = True
    for w in q.split()[::-1]: #read till last element
        if w in top_words:
            new_q = [w] + new_q
            new_memento = True
        elif w not in STOP_WORDS:
            if new_memento:
                new_q = ["memento"] + new_q
                new_memento = False
            if is_numeric(w):
                numbers_q = [w] + numbers_q
            else:
                surplus_q = [w] + surplus_q
        else:
            new_memento = True
        if len(new_q) == MAX_SEQUENCE_LENGTH:
            break
    new_q = " ".join(new_q) #use space to link all the list element
    return new_q, set(surplus_q), set(numbers_q)


def extract_features(df):
    q1s = np.array([""] * len(df), dtype=object)
    q2s = np.array([""] * len(df), dtype=object)
    features = np.zeros((len(df), 4))

    for i, (q1, q2) in enumerate(list(zip(df["question1"], df["question2"]))):
        q1s[i], surplus1, numbers1 = prepare(q1)
        q2s[i], surplus2, numbers2 = prepare(q2)
        features[i, 0] = len(surplus1.intersection(surplus2))
        features[i, 1] = len(surplus1.union(surplus2))
        features[i, 2] = len(numbers1.intersection(numbers2))
        features[i, 3] = len(numbers1.union(numbers2))

    return q1s, q2s, features




train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train["question1"] = train["question1"].fillna("").apply(preprocess)
train["question2"] = train["question2"].fillna("").apply(preprocess)

print("Creating the vocabulary of words occurred more than", MIN_WORD_OCCURRENCE)
all_questions = pd.Series(train["question1"].tolist() + train["question2"].tolist()).unique()
# Syntax explain: Convert a collection of text documents to a matrix of token counts
#                 \S: Matches any non-whitespace character; this is equivalent to the set [^\t\n\r\f\v].
# Words which occur more than 100 times in the train set are collected. The rest is considered as rare words and replaced !!!!!!
# When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold
#       find the pattern with '\S+' fisrt, then find the key of the words that occur at least MIN_WORD_OCCURRENCE times, apply this method to all_questions
vectorizer = CountVectorizer(lowercase=False, token_pattern="\S+", min_df=MIN_WORD_OCCURRENCE)
vectorizer.fit(all_questions) # Learn a vocabulary dictionary of all tokens in the raw documents.
top_words = set(vectorizer.vocabulary_.keys()) #vocabulary_: dict, A mapping of terms to feature indices.
top_words.add(REPLACE_WORD)

embeddings_index = get_embedding()
print("Words are not found in the embedding:", top_words - embeddings_index.keys())
top_words = embeddings_index.keys() #now the top_words are the words reported in GloVe
# now all the valid word is represented by GloVe

print("Train questions are being prepared for LSTM...")
q1s_train, q2s_train, train_q_features = extract_features(train)

# filters: list (or concatenation) of characters to filter out, such as punctuation.
#          Default: '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n' , includes basic punctuation, tabs, and newlines.
# texts = ["The sun is shining in June!","September is grey.","Life is beautiful in August.","I like it","This and other things?"]
# tokenizer.fit_on_texts(texts)
# print(tokenizer.word_index)
# tokenizer.texts_to_sequences(["June is beautiful and I like it!"])
#==>{'other': 17, 'shining': 5, 'june!': 6, 'and': 16, 'the': 3, 'things?': 18, 'august.': 11, 'life': 9, 'september': 7,
#    'sun': 4, 'i': 12, 'beautiful': 10, 'grey.': 8, 'in': 2, 'it': 14, 'like': 13, 'this': 15, 'is': 1}
#     !!!!!everytime you run the code, the order will be different, but the correspoding index of each word will stay the same
#==>[[1, 10, 16, 12, 13]]
tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(np.append(q1s_train, q2s_train)) #q1s+q2s in order
word_index = tokenizer.word_index

data_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_train), maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_train), maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(train["is_duplicate"])

nb_words = len(word_index) + 1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


print("Train features are being merged with NLP and Non-NLP features...")
train_nlp_features = pd.read_csv("data/nlp_features_train.csv")
train_non_nlp_features = pd.read_csv("data/non_nlp_features_train.csv")
train_magic_features = pd.read_csv("data/train_feature_graph_question_freq.csv")
# train_glove_to_word2vec = pd.read_csv("glove_embedding_train.csv")
# features_train = pd.concat([train_q_features, train_nlp_features, train_non_nlp_features,train_magic_features, ],axis=1)
features_train = np.hstack((train_q_features, train_nlp_features, train_non_nlp_features,train_magic_features))
# features_train = np.hstack((train_q_features, train_nlp_features, train_non_nlp_features,train_magic_features,train_glove_to_word2vec))


# !!!!!!!!! for test set !!!!!!!!! #
print("Same steps are being applied for test...")
test["question1"] = test["question1"].fillna("").apply(preprocess)
test["question2"] = test["question2"].fillna("").apply(preprocess)
q1s_test, q2s_test, test_q_features = extract_features(test)
test_data_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_test), maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_test), maxlen=MAX_SEQUENCE_LENGTH)
test_nlp_features = pd.read_csv("data/nlp_features_test.csv")
test_non_nlp_features = pd.read_csv("data/non_nlp_features_test.csv")
test_magic_features = pd.read_csv("data/test_feature_graph_question_freq.csv")
# test_glove_to_word2vec = pd.read_csv("glove_embedding_test.csv")
# features_test = pd.concat([test_q_features, test_nlp_features, test_non_nlp_features,test_magic_features, ],axis=1)
features_test = np.hstack((test_q_features, test_nlp_features, test_non_nlp_features,test_magic_features))
# features_test = np.hstack((test_q_features, test_nlp_features, test_non_nlp_features,test_magic_features,test_glove_to_word2vec))

#This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
#StratifiedKFold is a variation of k-fold which returns stratified folds: each set contains approximately the same percentage of samples of each target class as the complete set.
#here split to 10 set/model
# see: http://scikit-learn.org/stable/modules/cross_validation.html
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True)
model_count = 0

params = {}
params["objective"] = "binary:logistic"
params['eval_metric'] = 'logloss'
params["eta"] = 0.04
params["subsample"] = 0.8
params["min_child_weight"] = 2
params["colsample_bytree"] = 0.9
params["max_depth"] = 8
params["silent"] = 1
params["seed"] = 1632
y = pd.read_csv('data/train.csv')['is_duplicate']

totalpreds = []
kf = KFold(n=features_train.shape[0], n_folds=NUM_FOLDS, shuffle=True, random_state=2017)

# for idx_train, idx_val in skf.split(train["is_duplicate"], train["is_duplicate"]):
#     print("MODEL:", model_count)
#     x_train, x_eval = features_train[idx_train], features_train[idx_val]
#     y_train, y_eval = y[idx_train], y[idx_train]
#
#     d_train = xgb.DMatrix(x_train, label=y_train)
#     d_valid = xgb.DMatrix(x_eval, label=y_eval)
#     watchlist = [(d_valid, 'valid')]
#
#     bst = xgb.train(params, d_train, 40000, watchlist, early_stopping_rounds=100, verbose_eval=100)
#
#     print('start predicting on test...')
#     testpreds = bst.predict(xgb.DMatrix(features_test))
#     if model_count > 0:
#         totalpreds = totalpreds + testpreds
#     else:
#         totalpreds = testpreds
#     bst.save_model('xgb_model_fold_{}.model'.format(model_count))
#     model_count += 1

for index_train, index_eval in kf:
    x_train, x_eval = features_train[index_train], features_train[index_eval]
    y_train, y_eval = y[index_train], y[index_eval]
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_eval, label=y_eval)
    watchlist = [(d_valid, 'valid')]
    bst = xgb.train(params, d_train, 40000, watchlist, early_stopping_rounds=100, verbose_eval=100)
    print('start predicting on test...')
    testpreds = bst.predict(xgb.DMatrix(features_test))
    if model_count > 0:
        totalpreds = totalpreds + testpreds
    else:
        totalpreds = testpreds
    bst.save_model('model/xgb_model_fold_{}.model'.format(model_count))
    model_count += 1

totalpreds = totalpreds / model_count
test_id = pd.read_csv('data/test.csv')['test_id']
sub = pd.DataFrame()
sub['test_id'] = test_id
sub['is_duplicate'] = pd.Series(totalpreds)
# sub.to_csv('xgb_prediction.csv', index=False)

a = 0.16 / 0.37
b = (1 - 0.16) / (1 - 0.37)
trans = sub.is_duplicate.apply(lambda x: a * x / (a * x + b * (1 - x)))
sub['is_duplicate'] = trans
# sub.to_csv('xgb_prediction_trans.csv', index=False)
sub.to_csv('predictions/preds5.csv', index=False)
