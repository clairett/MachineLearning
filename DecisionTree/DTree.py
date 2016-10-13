from __future__ import division
import math
import operator

alpha = 1

class DTree(object):

    def __init__(self):
        self.cat = {}

    # For unigram
    def loadCorpus(self, filename):
        tag_dict = {}
        with open(filename, 'r') as infile:
            for line in infile:
                tags = line.strip().split()
                for tag in tags:
                    tag_dict[tag] = tag_dict.get(tag, 0)+1
        return tag_dict

    # For (history, data)
    def loadHistoryCorpus(self, filename):
        data = []
        with open(filename, 'r') as infile:
            for line in infile:
                tags = line.strip().split()
                #history = ['NO']
                for i, tag in enumerate(tags):
                    # print history
                    data_item = (['NO', 'NO'] + tags[:i], tag)
                    data.append(data_item)
                    #history.append(tag)
        return data


    def category1(self, entropy, tags, data):
        final_H1, final_H2 = [], []
        max_tag = ''
        max_MI = 0
        for tag in tags:
            H1, H2 = [], []
            for item in data:
                history = item[0]
                token = item[1]
                # print history[-1], tag
                if history[-1] == tag:
                    H1.append(item)
                else:
                    H2.append(item)
            MI = self.computeMI(entropy, H1, H2)
            if MI > max_MI:
                max_MI = MI
                max_tag = tag
                final_H1 = H1
                final_H2 = H2
            self.cat["1-"+tag] = MI
        return max_tag, max_MI, final_H1, final_H2

    def category2(self, entropy, tag_pairs, data):
        final_H1, final_H2 = [], []
        max_tag = ''
        max_MI = 0
        for tag_pair in tag_pairs:
            H1, H2 = [], []
            for item in data:
                history = item[0]
                token = item[1]
                if history[-1] == tag_pair[1] and history[-2] == tag_pair[0]:
                    H1.append(item)
                else:
                    H2.append(item)
            MI = self.computeMI(entropy, H1, H2)
            if MI > max_MI:
                max_MI = MI
                max_tag = tag_pair
                final_H1 = H1
                final_H2 = H2
            tags = ("2-", tag_pair[0], tag_pair[1])
            self.cat[tags] = MI
        return max_tag, max_MI, final_H1, final_H2

    def category3(self, entropy, tags, data):
        final_H1, final_H2 = [], []
        max_tag = ''
        max_MI = 0
        for tag in tags:
            H1, H2 = [], []
            for item in data:
                history = item[0]
                token = item[1]
                if tag in history:
                    H1.append(item)
                else:
                    H2.append(item)
            MI = self.computeMI(entropy, H1, H2)
            if MI > max_MI:
                max_MI = MI
                max_tag = tag
                final_H1 = H1
                final_H2 = H2
            self.cat["3-"+tag] = MI
        return max_tag, max_MI, final_H1, final_H2

    def category4(self, entropy, num, data):
        final_H1, final_H2 = [], []
        max_num = 0
        max_MI = 0
        for i in xrange(num):
            H1, H2 = [], []
            for item in data:
                history = item[0]
                token = item[1]
                count = self.getVerb(history)
                if count == num:
                    H1.append(item)
                else:
                    H2.append(item)
            MI = self.computeMI(entropy, H1, H2)
            if MI > max_MI:
                max_MI = MI
                max_num = i
                final_H1 = H1
                final_H2 = H2
            self.cat["4-"+str(i)] = MI
        return max_num, max_MI, final_H1, final_H2

    def getVerb(self, L):
        count = 0
        for token in L:
            if token.startswith('V'):
                count += 1
        return count

    def category5(self, entropy, num, data):
        final_H1, final_H2 = [], []
        max_num = 0
        max_MI = 0
        for i in xrange(num):
            H1, H2 = [], []
            for item in data:
                history = item[0]
                token = item[1]
                count = self.getPunc(history)
                if count == num:
                    H1.append(item)
                else:
                    H2.append(item)
            MI = self.computeMI(entropy, H1, H2)
            if MI > max_MI:
                max_MI = MI
                max_num = i
                final_H1 = H1
                final_H2 = H2
            self.cat["5-"+str(i)] = MI
        return max_num, max_MI, final_H1, final_H2

    def getPunc(self, L):
        puncs = [')', '(', ',', '.', ':',')']
        count = 0
        for token in L:
            if token in puncs:
                count += 1
        return count

    def computeMI(self, entropy, H1, H2):
        H1_dict = self.transform(H1)
        H2_dict = self.transform(H2)
        h1 = self.computeEntropy(H1_dict)
        h2 = self.computeEntropy(H2_dict)

        w1 = len(H1)/(len(H1)+len(H2))
        w2 = 1-w1
        return entropy - w1*h1 -w2*h2


    def transform(self, data_list):
        data_dict = {}
        for item in data_list:
            tag = item[1]
            data_dict[tag] = data_dict.get(tag, 0)+1
        return data_dict


    def computeEntropy(self, data_dict):
        entropy = 0
        all_counts = sum(data_dict.values())
        for key, value in data_dict.iteritems():
            p = value/all_counts
            entropy += -p*math.log(p, 2)
        return entropy


    def computeALL(self, train_tag_dict, test_tag_dict, train):
        ALL = 0
        if train:
            total_tags = sum(train_tag_dict.values())
            for tag, count in train_tag_dict.iteritems():
                ALL += math.log(count/total_tags,2)*count
        else:
            tags = {}
            total_tags = sum(test_tag_dict.values())
            total = 0
            for tag, count in test_tag_dict.iteritems():
                if tag in train_tag_dict:
                    total += train_tag_dict[tag]+alpha
                    tags[tag] = (train_tag_dict[tag]+alpha, count)
                else:
                    total += alpha
                    tags[tag] = (alpha, count)
            for tag, value in tags.iteritems():
                ALL += math.log(value[0]/total, 2)*value[1]
        return ALL

    def computeTrainALL(self, train_data1, train_data2):
        N1 = len(train_set1)
        N2 = len(train_set2)
        N = N1+N2
        ALL = 0
        train_set1 = self.transform(train_data1)
        train_set2 = self.transform(train_data2)
        for tag, count in train_set1:
            ALL += math.log(count/N1, 2)*count
        for tag, count in train_set2:
            ALL += math.log(count/N2, 2)*count
        return ALL/N

    # def computeTestALL(self, train1, train2, test1, test2):
        # train_set1 = self.transform(train1)
        # train_set2 = self.transform(train2)

        # test_set1 = self.transform(test1)
        # test_set2 = self.transform(test2)
        # tags1 = {}
        # tags2 = {}
        # for tag, count in test_set1:
            # if tag in train_set1:



    def splitTest1(self, tag, test_data):
        H1, H2= [], []
        for item in test_data:
            history = item[0]
            token = item[1]
            if tag in history:
                H1.append(item)
            else:
                H2.append(item)
        return H1, H2

    def computePP(self, ALL):
        return 2**(-ALL)


if __name__ == '__main__':
    tree = DTree()

    train_unigram_dict = tree.loadCorpus('./hw6-WSJ/hw6-WSJ-1.tags')
    entropy = tree.computeEntropy(train_unigram_dict)
    print entropy

    train_data = tree.loadHistoryCorpus('./hw6-WSJ/hw6-WSJ-1.tags')
    test_data = tree.loadHistoryCorpus('./hw6-WSJ/hw6-WSJ-2.tags')
    #print train_data

    tags = list(train_unigram_dict.keys())+['NO']
    print tags
    tag_pairs = []
    for tag1 in tags:
        for tag2 in tags:
            pair = (tag1, tag2)
            tag_pairs.append(pair)

    max_tag, max_MI, max_H1, max_H2= tree.category1(entropy, tags, train_data)
    print "Category 1: Tag %s gives max MI %f" % (max_tag, max_MI)

    tree.category2(entropy, tag_pairs, train_data)
    # print "Category 2: Tag %s gives max MI %f" % (max_tag, max_MI)

    # max_tag, max_MI, max_H1, max_H2= tree.category3(entropy, tags, train_data)
    # print "Category 3: Tag %s gives max MI %f" % (max_tag, max_MI)

    # max_num, max_MI, max_H1, max_H2= tree.category4(entropy, 5, train_data)
    # print "Category 4: Num %s gives max MI %f" % (max_num, max_MI)

    # max_num, max_MI, max_H1, max_H2= tree.category5(entropy, 10, train_data)
    # print "Category 5: Num %s gives max MI %f" % (max_num, max_MI)

    # max_tag21, max_MI21, max_H11, max_H12 = tree.category1(entropy, tags, max_H1)
    # max_tag22, max_MI22, max_H21, max_H22 = tree.category1(entropy, tags, max_H2)
    # print "Category 1: Tag %s gives max MI %f" % (max_tag21, max_MI21)
    # print "Category 1: Tag %s gives max MI %f" % (max_tag22, max_MI22)

    sorted_cat = sorted(tree.cat.items(), key=operator.itemgetter(1))
    for i in xrange(50):
        print sorted_cat[i]


    reversed_sorted_cat = sorted(tree.cat.items(), key=operator.itemgetter(1), reverse=True)
    for i in xrange(50):
        print reversed_sorted_cat[i]


    # trainset1 = tree.transform(max_H1)
    # trainset2 = tree.trainform(max_H2)

    # train_ALL = tree.computeALL(trainset1, )






