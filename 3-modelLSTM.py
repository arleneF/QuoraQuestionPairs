# Reference: word embedding:
# http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/
# https://yq.aliyun.com/articles/221681
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
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
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=""

np.random.seed(0) #makes the random numbers predictable
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
features_test = np.hstack((test_q_features, test_nlp_features, test_non_nlp_features,test_magic_features))
# features_test = np.hstack((test_q_features, test_nlp_features, test_non_nlp_features,test_magic_features,test_glove_to_word2vec))

#This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
#StratifiedKFold is a variation of k-fold which returns stratified folds: each set contains approximately the same percentage of samples of each target class as the complete set.
#here split to 10 set/model
# see: http://scikit-learn.org/stable/modules/cross_validation.html
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True)
model_count = 0

for idx_train, idx_val in skf.split(train["is_duplicate"], train["is_duplicate"]):
    print("MODEL:", model_count)
    data_1_train = data_1[idx_train]
    data_2_train = data_2[idx_train]
    labels_train = labels[idx_train]
    f_train = features_train[idx_train]

    data_1_val = data_1[idx_val]
    data_2_val = data_2[idx_val]
    labels_val = labels[idx_val]
    f_val = features_train[idx_val]

    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(150, dropout=0.15,recurrent_dropout=0.15)
    # lstm_layer = LSTM(75, recurrent_dropout=0.2)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    features_input = Input(shape=(f_train.shape[1],), dtype="float32")
    features_dense = BatchNormalization()(features_input)
    features_dense = Dense(200, activation="relu")(features_dense)
    features_dense = Dropout(0.2)(features_dense)

    addition = add([x1, y1])
    minus_y1 = Lambda(lambda x: -x)(y1)
    merged = add([x1, minus_y1])
    merged = multiply([merged, merged])
    merged = concatenate([merged, addition])
    merged = Dropout(0.4)(merged)

    merged = concatenate([merged, features_dense])
    merged = BatchNormalization()(merged)
    merged = GaussianNoise(0.1)(merged)

    merged = Dense(150, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    out = Dense(1, activation="sigmoid")(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input, features_input], outputs=out)
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",metrics=['acc']) #originally is nadam
    #https://github.com/fchollet/keras/issues/605
    early_stopping = EarlyStopping(monitor="val_loss", patience=5) #Stop training when a monitored quantity has stopped improving.
    best_model_path = "model/best_model" + str(model_count) + ".h5"
    best_model_path_whole_model= "model/whole_model_best_model" + str(model_count) + ".h5"
    model.save(best_model_path_whole_model)
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True) #Save the model after every epoch.

    hist = model.fit([data_1_train, data_2_train, f_train], labels_train,
                     validation_data=([data_1_val, data_2_val, f_val], labels_val),
                     epochs=15, batch_size=BATCH_SIZE, shuffle=True,
                     callbacks=[early_stopping, model_checkpoint], verbose=1)

    model.load_weights(best_model_path)
    print(model_count, "validation loss:", min(hist.history["val_loss"]))

    preds = model.predict([test_data_1, test_data_2, features_test], batch_size=BATCH_SIZE, verbose=1)

    submission = pd.DataFrame({"test_id": test["test_id"], "is_duplicate": preds.ravel()})
    submission.to_csv("predictions/preds" + str(model_count) + ".csv", index=False)

    model_count += 1
