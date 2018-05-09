import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import feat_gen
# import importlib; importlib.reload(feat_gen)
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from gensim.models import KeyedVectors
from tqdm import tqdm, tqdm_pandas
from nltk.corpus import stopwords
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
stop_words = stopwords.words('english')
import re

def preprocess(x):
    x = str(x).lower()
    #str.replace(old, new[, max])
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
							.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
							.replace("n't", " not").replace("what's", "what is").replace("it's", "it is").replace("wasn't","was not")\
                            .replace("hasn't","has not").replace("wouldn't","would not").replace("isn't", "is not") \
							.replace("shouldn't","should not").replace("weren't","were not").replace("couldn't","could not") \
							.replace("didn't","did not").replace("aren't","are not").replace("needn't","need not") \
							.replace("'ve", " have").replace("i'm", "i am").replace("'re", " are").replace("mighn't","might not")\
							.replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
							.replace("bodylanguage","body language").replace("englisn","english") \
							.replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
							.replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")\
							.replace("e-mai", " email ").replace("'d","would")
    #re.sub(pattern, repl, string, count=0, flags=0)
    #Reference: https://lzone.de/examples/Python%20re.sub
    x = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', x)
    x = re.sub(r"([0-9]+)000000", r"\1m", x) #change to million
    x = re.sub(r"([0-9]+)000", r"\1k", x) #change to 0k-9k, this should be after million
    #there are some typo, need to be corrected
    x = re.sub(r" e g ", " eg ", x)
    x = re.sub(r" b g ", " bg ", x)
    x = re.sub(r"\0s", "0", x)
    x = re.sub(r" 9 11 ", "911", x)
    x = re.sub(r"quikly", "quickly", x)
    x = re.sub(r"imrovement", "improvement", x)
    x = re.sub(r"intially", "initially", x)
    x = re.sub(r" dms ", "direct messages ", x)
    x = re.sub(r"demonitization", "demonetization", x)
    x = re.sub(r"kms", " kilometers ", x)
    x = re.sub(r" cs ", " computer science ", x)
    x = re.sub(r" upvotes ", " up votes ", x)
    x = re.sub(r"\0rs ", " rs ", x)
    x = re.sub(r"calender", "calendar", x)
    x = re.sub(r"programing", "programming", x)
    x = re.sub(r"bestfriend", "best friend", x)
    x = re.sub(r"iii", "3", x) #III
    x = re.sub(r" j k ", " jk ", x)
    return x

train =  pd.read_csv('data/train.csv')
test =  pd.read_csv('data/test.csv')
train["question1"] = train["question1"].fillna("").apply(preprocess)
train["question2"] = train["question2"].fillna("").apply(preprocess)
test["question1"] = test["question1"].fillna("").apply(preprocess)
test["question2"] = test["question2"].fillna("").apply(preprocess)


# model = KeyedVectors.load_word2vec_format('./word2Vec_models/glove_w2vec.txt')
model = KeyedVectors.load_word2vec_format('glove_word2vec_convert.txt',binary=False)
# model = KeyedVectors.load_word2vec_format('glove_840B.300d.txt',binary=False)
# if use this line, the error msg:
# Traceback (most recent call last):
#   File "2.2-glove_embedding.py", line 64, in <module>
#     model = KeyedVectors.load_word2vec_format('glove.840B.300d.txt',binary=False)
#   File "/usr/local/lib/python3.5/dist-packages/gensim/models/keyedvectors.py", line 204, in load_word2vec_format
#     vocab_size, vector_size = (int(x) for x in header.split())  # throws for invalid file format
#   File "/usr/local/lib/python3.5/dist-packages/gensim/models/keyedvectors.py", line 204, in <genexpr>
#     vocab_size, vector_size = (int(x) for x in header.split())  # throws for invalid file format
# ValueError: invalid literal for int() with base 10: ','

# therefore, need:
# python -m gensim.scripts.glove2word2vec --input 'glove.840B.300d.txt' --output 'glove_word2vec_convert.txt'
# https://github.com/jroakes/glove-to-word2vec
# https://github.com/3Top/word2vec-api/issues/6
# https://radimrehurek.com/gensim/scripts/glove2word2vec.html

train_df = pd.DataFrame()
test_df = pd.DataFrame()

def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    norm=np.sqrt((v**2).sum())
    if norm>0:
        return v / np.sqrt((v ** 2).sum())
    else:
        return None


question1_vectors_train = np.zeros((train.shape[0], 300))
question2_vectors_train = np.zeros((train.shape[0], 300))
error_count_train = 0
for i, q in tqdm(enumerate(train.question1.values)):
    question1_vectors_train[i, :] = sent2vec(q)
for i, q in tqdm(enumerate(train.question2.values)):
    question2_vectors_train[i, :] = sent2vec(q)
train_df['cosine_distance2'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors_train),np.nan_to_num(question2_vectors_train))]
train_df['cityblock_distance2'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors_train),np.nan_to_num(question2_vectors_train))]
train_df['jaccard_distance2'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors_train),np.nan_to_num(question2_vectors_train))]
train_df['canberra_distance2'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors_train), np.nan_to_num(question2_vectors_train))]
train_df['euclidean_distance2'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors_train), np.nan_to_num(question2_vectors_train))]
train_df['minkowski_distance2'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors_train), np.nan_to_num(question2_vectors_train))]
train_df['braycurtis_distance2'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors_train), np.nan_to_num(question2_vectors_train))]
train_df['skew_q1vec2'] = [skew(x) for x in np.nan_to_num(question1_vectors_train)]
train_df['skew_q2vec2'] = [skew(x) for x in np.nan_to_num(question2_vectors_train)]
train_df['kur_q1vec2'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors_train)]
train_df['kur_q2vec2'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors_train)]



question1_vectors_test = np.zeros((test.shape[0], 300))
question2_vectors_test = np.zeros((test.shape[0], 300))
error_count_test = 0
for i, q in tqdm(enumerate(test.question1.values)):
    question1_vectors_test[i, :] = sent2vec(q)
for i, q in tqdm(enumerate(test.question2.values)):
    question2_vectors_test[i, :] = sent2vec(q)
test_df['cosine_distance2'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors_test),np.nan_to_num(question2_vectors_test))]
test_df['cityblock_distance2'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors_test),np.nan_to_num(question2_vectors_test))]
test_df['jaccard_distance2'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors_test),np.nan_to_num(question2_vectors_test))]
test_df['canberra_distance2'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors_test), np.nan_to_num(question2_vectors_test))]
test_df['euclidean_distance2'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors_test),np.nan_to_num(question2_vectors_test))]
test_df['minkowski_distance2'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors_test),np.nan_to_num(question2_vectors_test))]
test_df['braycurtis_distance2'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors_test),np.nan_to_num(question2_vectors_test))]
test_df['skew_q1vec2'] = [skew(x) for x in np.nan_to_num(question1_vectors_test)]
test_df['skew_q2vec2'] = [skew(x) for x in np.nan_to_num(question2_vectors_test)]
test_df['kur_q1vec2'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors_test)]
test_df['kur_q2vec2'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors_test)]


# del question1_vectors_train, question2_vectors_train
# del question1_vectors_test, question2_vectors_test

train_df.to_csv('data/glove_embedding_train.csv', index=False)
test_df.to_csv('data/glove_embedding_test.csv', index=False)
