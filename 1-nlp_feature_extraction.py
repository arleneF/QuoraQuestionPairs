import re
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import distance
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=""
SAFE_DIV = 0.0001
STOP_WORDS = stopwords.words("english")

# def preprocess(x):
#     x = str(x).lower()
#     x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
#                            .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
#                            .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
#                            .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
#                            .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
#                            .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
#                            .replace("€", " euro ").replace("'ll", " will")
#     x = re.sub(r"([0-9]+)000000", r"\1m", x)
#     x = re.sub(r"([0-9]+)000", r"\1k", x)
#     return x

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

def get_token_features(q1, q2):
    q1_tokens={}
    q2_tokens={}
    token_features = [0.0]*10

    #split by space
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    # if either one is empty
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    #word share feature from benchmark model
    # for word in q1_tokens:
    #     if word not in STOP_WORDS:
    #         q1_words[word]=1;
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS]) #very common engilish word like also, am, and ....
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    #this means all the sentence is nothing but stopwords, which is more likely a computer-generted chaff
    # if len(q1_words)==0 or len(q2_words)==0:
    #     return 0;
    #if use these above TWO LINES, it will generate error:
    # Traceback (most recent call last):
    #     File "1-nlp_feature_extraction.py", line 153, in <module>
    #         train_df = extract_features(train_df)
    #     File "1-nlp_feature_extraction.py", line 129, in extract_features
    #         df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    #     File "1-nlp_feature_extraction.py", line 129, in <lambda>
    #         df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    # TypeError: 'int' object is not subscriptable


    # common_word_count_q1=[w for w in q1_words.keys() if w in q2_words]
    # common_word_count_q2=[w for w in q2_words.keys() if w in q1_words]
    # common_word_count=(len(common_word_count_q1)+len(common_word_count_q2))/(len(q1_words)+len(q2_words))
    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features

def get_longest_substr_ratio(a, b):
    # lcsubstrings : find the longest common substrings in two sequences
    # lcsubstrings("sedentar", "dentist") ==>{'dent'}
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

def extract_features(df):
    df["question1"] = df["question1"].fillna("").apply(preprocess)
    df["question2"] = df["question2"].fillna("").apply(preprocess)

    print("token features...")
    token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
    df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    df["csc_min"]       = list(map(lambda x: x[2], token_features))
    df["csc_max"]       = list(map(lambda x: x[3], token_features))
    df["ctc_min"]       = list(map(lambda x: x[4], token_features))
    df["ctc_max"]       = list(map(lambda x: x[5], token_features))
    df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
    df["mean_len"]      = list(map(lambda x: x[9], token_features))

    print("fuzzy features..")
    df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_Qratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_Wratio"]            = df.apply(lambda x: fuzz.WRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df['fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(x["question1"], x["question2"]), axis=1)
    df['fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    return df

print("Extracting features for train:")
train_df = pd.read_csv("data/train.csv")
train_df = extract_features(train_df)
train_df.drop(["id", "qid1", "qid2", "question1", "question2", "is_duplicate"], axis=1, inplace=True) #the origin colomn is dropped, only leave preprocessed result
train_df.to_csv("data/nlp_features_train.csv", index=False)

print("Extracting features for test:")
test_df = pd.read_csv("data/test.csv")
test_df = extract_features(test_df)
test_df.drop(["test_id", "question1", "question2"], axis=1, inplace=True)
test_df.to_csv("data/nlp_features_test.csv", index=False)
