##############################
#                            #
#   Naive Bayes Classifier   #
#                            #
##############################

__author__ = 'tiantian'


import collections
import math
import os

# union negative words and positive words
# store the union in sentiment_words
sentiment_words = []
filenames = ["opinion-lexicon-English/negative-words.txt", "opinion-lexicon-English/positive-words.txt"]
for sentiment_file in filenames:
    for line in open(sentiment_file):
        li = line.strip()
        if not li.startswith(";") and li != "":
            sentiment_words.append(li)

D = len(sentiment_words) # the dimension of the feature vector

# construct feature matrix for neg words
instance_neg = 0
train_feature_neg = [[0 for j in range(D)] for i in range(800) ]
for neg_file in os.listdir("txt_sentoken/neg/"):
    words = []

    # get words from a single file
    for line in open("txt_sentoken/neg/"+neg_file):
        line = line.strip()
        line = line.lower()
        words += line.split(" ")

    for w in words:
        if w in sentiment_words:
            j = sentiment_words.index(w)
            train_feature_neg[instance_neg][j] = 1

    instance_neg += 1
    if instance_neg == 800:
        break

# construct feature matrix for pos words
instance_pos = 0
train_feature_pos = [[0 for j in range(D)] for i in range(800) ]
for pos_file in os.listdir("txt_sentoken/pos/"):
    words = []
    for line in open('txt_sentoken/pos/'+pos_file):
        line = line.strip()
        line = line.lower()
        words += line.split(" ")

    for w in words:
        if w in sentiment_words:
            j = sentiment_words.index(w)
            train_feature_pos[instance_pos][j] = 1
    instance_pos += 1
    if instance_pos == 800:
        break

# get the counts for token with tag
token_neg_count = collections.defaultdict(int)
token_pos_count = collections.defaultdict(int)

# compute count(token, neg)
for i in range(800):
    for j in range(D):
        if train_feature_neg[i][j] == 1:
            token_neg_count[sentiment_words[j]] += 1


# compute count(token, pos)
for i in range(800):
    for j in range(D):
        if train_feature_pos[i][j] == 1:
            token_pos_count[sentiment_words[j]] += 1


prob_token_neg = dict()
prob_token_pos = dict()

# compute prob(token|neg)
for token in sentiment_words:
    if token in token_neg_count.keys():
        prob_token_neg[token] = (token_neg_count[token]*1.0 + 0.1) / (800 + 0.2)
    else:
        prob_token_neg[token] = 0.1/(800 + 0.2)

# compute prob(token|pos)
for token in sentiment_words:
    if token in token_pos_count.keys():
        prob_token_pos[token] = (token_pos_count[token]*1.0 + 0.1) / (800 + 0.2)
    else:
        prob_token_pos[token] = 0.1/(800 + 0.2)


# construct feature vectors for test data with neg tag
test_feature_neg = [[0 for j in range(D)] for i in range(200) ]
instance_neg = 0
for test_neg in os.listdir("txt_sentoken/neg_test/"):
    words = []
    for line in open("txt_sentoken/neg_test/"+test_neg):
        line = line.strip()
        line = line.lower()
        words += line.split(" ")

    for w in words:
        if w in sentiment_words:
            j = sentiment_words.index(w)
            test_feature_neg[instance_neg][j] = 1

    instance_neg += 1

# construct feature vectors for test data with pos tag
test_feature_pos = [[0 for j in range(D)] for i in range(200) ]
instance_pos = 0
for test_pos in os.listdir("txt_sentoken/pos_test/"):
    words = []
    for line in open("txt_sentoken/pos_test/"+test_pos):
        line = line.strip()
        line = line.lower()
        words += line.split(" ")

    for w in words:
        if w in sentiment_words:
            j = sentiment_words.index(w)
            test_feature_pos[instance_pos][j] = 1

    instance_pos += 1

# learn for train
train_ret_neg = 0
for i in range(800):
    prob_neg = 0
    prob_pos = 0

    for j in range(D):
        if train_feature_neg[i][j] == 1:
            prob_neg += math.log(prob_token_neg[sentiment_words[j]])
            prob_pos += math.log(prob_token_pos[sentiment_words[j]])
        else:
            prob_neg += math.log((1-prob_token_neg[sentiment_words[j]]))
            prob_pos += math.log((1-prob_token_pos[sentiment_words[j]]))

    if prob_neg >= prob_pos:
        train_ret_neg += 1

train_ret_pos = 0
for i in range(800):
    prob_neg = 0
    prob_pos = 0

    for j in range(D):
        if train_feature_pos[i][j] == 1:
            prob_neg += math.log(prob_token_neg[sentiment_words[j]])
            prob_pos += math.log(prob_token_pos[sentiment_words[j]])
        else:
            prob_neg += math.log((1-prob_token_neg[sentiment_words[j]]))
            prob_pos += math.log((1-prob_token_pos[sentiment_words[j]]))

    if prob_pos >= prob_neg:
        train_ret_pos += 1

print train_ret_neg
print train_ret_pos

train_accuracy = (train_ret_neg + train_ret_pos) * 1.0 / 16

# learn for test
ret_neg = 0
for i in range(200):
    prob_neg = 0
    prob_pos = 0

    for j in range(D):
        if test_feature_neg[i][j] == 1:
            prob_neg += math.log(prob_token_neg[sentiment_words[j]])
            prob_pos += math.log(prob_token_pos[sentiment_words[j]])
        else:
            prob_neg += math.log((1-prob_token_neg[sentiment_words[j]]))
            prob_pos += math.log((1-prob_token_pos[sentiment_words[j]]))

    if prob_neg >= prob_pos:
        ret_neg += 1


ret_pos = 0
for i in range(200):
    prob_neg = 0
    prob_pos = 0

    for j in range(D):
        if test_feature_pos[i][j] == 1:
            prob_neg += math.log(prob_token_neg[sentiment_words[j]])
            prob_pos += math.log(prob_token_pos[sentiment_words[j]])
        else:
            prob_neg += math.log((1-prob_token_neg[sentiment_words[j]]))
            prob_pos += math.log((1-prob_token_pos[sentiment_words[j]]))

    if prob_pos >= prob_neg:
        ret_pos += 1


test_accuracy = (ret_neg + ret_pos)*1.0 / 4


print "Train Accuracy: " + str(train_accuracy) + "%"
print "Test Accuracy: " + str(test_accuracy) + "%"


# print top 10 words
sorted_neg = sorted(prob_token_neg.iteritems(), key = lambda x:x[1], reverse=True)
sorted_pos = sorted(prob_token_pos.iteritems(), key = lambda x:x[1], reverse=True)

print "TOP 10 HIGHEST NEGATIVE WORDS"
for i in range(10):
    print sorted_neg[i]

print "TOP 10 HIGHEST POSITIVE WORDS"
for i in range(10):
    print sorted_pos[i]

