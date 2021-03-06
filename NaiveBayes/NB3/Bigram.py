########################################################
#                                                      #
#   Naive Bayes with Negation Handling and Bi-grams    #
#                                                      #
########################################################


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
            sentiment_words.append("not_"+li)
            sentiment_words.append("extremely_"+li)
            sentiment_words.append("quite_"+li)
            sentiment_words.append("just_"+li)
            sentiment_words.append("almost_"+li)
            sentiment_words.append("very_"+li)
            sentiment_words.append("too_"+li)
            sentiment_words.append("enough_"+li)

D = len(sentiment_words) # the dimension of the feature vector

# construct feature matrix for neg words
count1 = 0
train_feature_neg = [[0 for j in range(D)] for i in range(800) ]
for neg_file in os.listdir("txt_sentoken/neg/"):
    words = []

    # get words from a single file
    for line in open("txt_sentoken/neg/"+neg_file):
        line = line.strip()
        words += line.split(" ")

    # bi-gram words
    for i in range(len(words)):
        if words[i] == "not" and i < len(words) - 1:
            words[i+1] = "not_"+words[i+1]

        if words[i] == "extremely" and i < len(words) - 1:
            words[i+1] = "extremely_"+words[i+1]

        if words[i] == "quite" and i < len(words) - 1:
            words[i+1] = "quite_"+words[i+1]

        if words[i] == "just" and i < len(words) - 1:
            words[i+1] = "just_"+words[i+1]

        if words[i] == "almost" and i < len(words) - 1:
            words[i+1] = "almost_"+words[i+1]

        if words[i] == "very" and i < len(words) - 1:
            words[i+1] = "very_"+words[i+1]

        if words[i] == "too" and i < len(words) - 1:
            words[i+1] = "too_"+words[i+1]

        if words[i] == "enough" and i < len(words) - 1:
            words[i+1] = "enough_"+words[i+1]

    for w in words:
        if w in sentiment_words:
            j = sentiment_words.index(w)
            train_feature_neg[count1][j] = 1

    count1 += 1
    if count1 == 800:
        break


# construct feature matrix for pos words
count2 = 0
train_feature_pos = [[0 for j in range(D)] for i in range(800) ]
for pos_file in os.listdir("txt_sentoken/pos/"):
    words = []
    for line in open('txt_sentoken/pos/'+pos_file):
        line = line.strip()
        words += line.split(" ")

    # bi-gram words
    for i in range(len(words)):
        if words[i] == "not" and i < len(words) - 1:
            words[i+1] = "not_"+words[i+1]

        if words[i] == "extremely" and i < len(words) - 1:
            words[i+1] = "extremely_"+words[i+1]

        if words[i] == "quite" and i < len(words) - 1:
            words[i+1] = "quite_"+words[i+1]

        if words[i] == "just" and i < len(words) - 1:
            words[i+1] = "just_"+words[i+1]

        if words[i] == "almost" and i < len(words) - 1:
            words[i+1] = "almost_"+words[i+1]

        if words[i] == "very" and i < len(words) - 1:
            words[i+1] = "very_"+words[i+1]

        if words[i] == "too" and i < len(words) - 1:
            words[i+1] = "too_"+words[i+1]

        if words[i] == "enough" and i < len(words) - 1:
            words[i+1] = "enough_"+words[i+1]

    for w in words:
        if w in sentiment_words:
            j = sentiment_words.index(w)
            train_feature_pos[count2][j] = 1

    count2 += 1
    if count2 == 800:
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

    # bi-gram words
    for i in range(len(words)):
        if words[i] == "not" and i < len(words) - 1:
            words[i+1] = "not_"+words[i+1]

        if words[i] == "extremely" and i < len(words) - 1:
            words[i+1] = "extremely_"+words[i+1]

        if words[i] == "quite" and i < len(words) - 1:
            words[i+1] = "quite_"+words[i+1]

        if words[i] == "just" and i < len(words) - 1:
            words[i+1] = "just_"+words[i+1]

        if words[i] == "almost" and i < len(words) - 1:
            words[i+1] = "almost_"+words[i+1]

        if words[i] == "very" and i < len(words) - 1:
            words[i+1] = "very_"+words[i+1]

        if words[i] == "too" and i < len(words) - 1:
            words[i+1] = "too_"+words[i+1]

        if words[i] == "enough" and i < len(words) - 1:
            words[i+1] = "enough_"+words[i+1]

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

    # bi-gram words
    for i in range(len(words)):
        if words[i] == "not" and i < len(words) - 1:
            words[i+1] = "not_"+words[i+1]

        if words[i] == "extremely" and i < len(words) - 1:
            words[i+1] = "extremely_"+words[i+1]

        if words[i] == "quite" and i < len(words) - 1:
            words[i+1] = "quite_"+words[i+1]

        if words[i] == "just" and i < len(words) - 1:
            words[i+1] = "just_"+words[i+1]

        if words[i] == "almost" and i < len(words) - 1:
            words[i+1] = "almost_"+words[i+1]

        if words[i] == "very" and i < len(words) - 1:
            words[i+1] = "very_"+words[i+1]

        if words[i] == "too" and i < len(words) - 1:
            words[i+1] = "too_"+words[i+1]

        if words[i] == "enough" and i < len(words) - 1:
            words[i+1] = "enough_"+words[i+1]

    for w in words:
        if w in sentiment_words:
            j = sentiment_words.index(w)
            test_feature_pos[instance_pos][j] = 1

    instance_pos += 1


# learn for train
train_ret_pos = 0
for i in range(800):
    prob_neg = 0
    prob_pos = 0

    for j in range(D):
        if train_feature_pos[i][j] == 1:
            prob_neg += math.log(prob_token_neg[sentiment_words[j]])
            prob_pos += math.log(prob_token_pos[sentiment_words[j]])
        else:
            prob_neg += math.log(1-prob_token_neg[sentiment_words[j]])
            prob_pos += math.log(1-prob_token_pos[sentiment_words[j]])

    if prob_pos >= prob_neg:
        train_ret_pos += 1

train_ret_neg = 0
for i in range(800):
    prob_neg = 0
    prob_pos = 0

    for j in range(D):
        if train_feature_neg[i][j] == 1:
            prob_neg += math.log(prob_token_neg[sentiment_words[j]])
            prob_pos += math.log(prob_token_pos[sentiment_words[j]])
        else:
            prob_neg += math.log(1-prob_token_neg[sentiment_words[j]])
            prob_pos += math.log(1-prob_token_pos[sentiment_words[j]])

    if prob_neg >= prob_pos:
        train_ret_neg += 1

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
            prob_neg += math.log(1-prob_token_neg[sentiment_words[j]])
            prob_pos += math.log(1-prob_token_pos[sentiment_words[j]])

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
            prob_neg += math.log(1-prob_token_neg[sentiment_words[j]])
            prob_pos += math.log(1-prob_token_pos[sentiment_words[j]])

    if prob_pos >= prob_neg:
        ret_pos += 1


test_accuracy = (ret_neg + ret_pos)*1.0 / 4

print "Train Accuracy: " + str(train_accuracy) + "%"
print "Test Accuracy: " + str(test_accuracy) + "%"



