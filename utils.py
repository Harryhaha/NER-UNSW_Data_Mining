__author__ = "Harry"

import re
from sklearn import linear_model
import pickle
import sys
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold
from nltk.corpus import stopwords
import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.metrics import f1_score
# import matplotlib.pyplot as plt

wnl = WordNetLemmatizer()
stopwords = stopwords.words('english')



"""
Build the Gazatteer
"""

stopwords_knowledge = defaultdict(int)
for item in stopwords:
    stopwords_knowledge[item] = 1

file = open('Gazatteer.txt', 'r', encoding='utf8')
external_knowledge = defaultdict(int)
for line in file.readlines():
    line = line.strip()
    if len(line) == 0: continue
    line = line.lower()  # Transfer to lowercase
    external_knowledge[line] = 1
file.close()

file = open('OneDistanceTagsForTitleWithHighProb.txt', 'r', encoding='utf8')
POS_oneDistanceTag_TITLE_list = []
POS_twoDistanceTag_TITLE_list = []

for line in file.readlines():
    line = line.strip()
    if len(line) == 0: continue
    POS_oneDistanceTag_TITLE_list.append( line )
file.close()

file = open('TwoDistanceTagsForTitleWithHighProb.txt', 'r', encoding='utf8')
for line in file.readlines():
    line = line.strip()
    if len(line) == 0: continue
    POS_twoDistanceTag_TITLE_list.append( line )
file.close()





"""
weight adjusted for each feature
"""

W_token_itself = 3
W_external_knowledge = 3
W_external_stopwords = 2
W_as_title_of = 2
W_serve_as = 2
W_become = 2
W_not_noun = 1
W_before_after_text = 3
W_before_before_text = 1
W_after_after_text = 1
W_prev_curr_next_tags = 0.5
W_surrounding_two_curr_tags = 0.5




"""
Auxiliary functions called by core functions for feature extraction
"""

# Use "as [DT]? [NNP/NN] (CC [DT]? [NNP/NN])* of" as one feature
def F_as_TITLE_of(sent_list, NN_index):
    # Now NN_index is already NNP/NN
    exclude_word_list = ["member", "tool", "improvement", "confirmation", "student", "rite",
                         "part", "leader", "center", "candidate", "blusukan", "sign", "result",
                         "guest", "ally", "bureau", "successor", "facilitator", "stronghold",
                         "head", "reason", "lack", "convenor", "core"]
    if wnl.lemmatize( sent_list[NN_index][0].lower() ) in exclude_word_list:
        return False

    i = NN_index-1
    flag_not_feature_anymore = False
    while i >= 0 and \
            (sent_list[i][1] == "NNP" or sent_list[i][1] == "NN" or
                     sent_list[i][1] == "CC" or sent_list[i][1] == "DT"):
        if sent_list[i][1] == "CC" and sent_list[i][0].lower() != "and":
            flag_not_feature_anymore = True
            break
        if sent_list[i][1] != "CC" and sent_list[i][1] != "DT" and \
                (sent_list[i][0] in stopwords or sent_list[i][0] in exclude_word_list):
            flag_not_feature_anymore = True
            break
        i = i-1
    if not (i >= 0 and flag_not_feature_anymore is False and sent_list[i][0].lower() == "as"):
        return False

    i = NN_index+1
    while i < len(sent_list) and \
            (sent_list[i][1] == "NNP" or sent_list[i][1] == "NN" or
                     sent_list[i][1] == "CC" or sent_list[i][1] == "DT"):
        if sent_list[i][1] == "CC" and sent_list[i][0].lower() != "and":
            flag_not_feature_anymore = True
            break
        if sent_list[i][1] != "CC" and sent_list[i][1] != "DT" and \
                (sent_list[i][0] in stopwords or sent_list[i][0] in exclude_word_list):
            flag_not_feature_anymore = True
            break
        i = i+1
    if not(i <= (len(sent_list)-1) and flag_not_feature_anymore is False
           and sent_list[i][0].lower() == "of"):
        return False
    return True

# Use "serve[s]? as [PRP$/DT/JJ]? ([NN/NNP])+" as one feature
def F_serve_as(sent_list, NN_index):
    flag_not_feature_anymore = False
    i = NN_index-1
    while i >= 0 and (sent_list[i][1] == "NNP" or sent_list[i][1] == "NN"):
        i -= 1
    if i <= 0: return False

    if sent_list[i][1] == "JJ":
        while i >= 0 and sent_list[i][1] == "JJ":
            i -= 1
    if i <= 0: return False

    if sent_list[i][1] == "DT" or sent_list[i][1] == "PRP$": i -= 1
    if i < 1: return False

    if sent_list[i][0].lower() == "as" and \
            (sent_list[i-1][0].lower()=="serve" or sent_list[i-1][0].lower()=="serves"):
        return True
    else: return False





"""
Core functions for feature extraction used for training/testing
"""

def BECOME_feature(token_list, sentence_list, weight_become, index, feature_dict):
    pattern_become = re.compile(r'\bbecame|become\b', re.IGNORECASE)
    if token_list[1] == "NN" or token_list[1] == "NNP":
        if index >= 1 and pattern_become.match(sentence_list[index - 1][0]):
            feature_dict["BECOME"] = weight_become
        elif index >= 2 and pattern_become.match(sentence_list[index - 2][0]):
            if sentence_list[index - 1][1] == "DT" or sentence_list[index - 1][1] == "JJ":
                feature_dict["BECOME"] = weight_become
    return feature_dict


def token_POS_combine_feature(token_text, token_list, feature_dict):
    # feature_dict[ token_text ] = weight_token_itself
    token_POS = token_text + "+" + token_list[1]
    feature_dict[token_POS] = W_token_itself
    return feature_dict

def gazatteer_feature(token_text, external_knowledge, stopwords_knowledge, feature_dict):
    if token_text in external_knowledge:
        feature_dict["EXTERNALKNOWLEDGE"] = W_external_knowledge
    if token_text in stopwords_knowledge:
        feature_dict["EXTERNALSTOPWORDS"] = W_external_stopwords
    return feature_dict


def not_noun_feature(token_list, weight_not_noun, feature_dict):
    if token_list[1] != "NN" or token_list[1] != "NNP" or \
                    token_list[1] != "NNS" or token_list[1] != "NNPS":
        feature_dict["NOTNOUN"] = weight_not_noun
    return feature_dict


def as_of_feature(token_list, feature_as_TITLE_of, sentence_list, index,
                  weight_as_title_of, feature_dict):
    if token_list[1] == "NNP" or token_list[1] == "NN":
        if (feature_as_TITLE_of(sentence_list, index)) is True:
            feature_dict["ASTITLEOF"] = weight_as_title_of
    return feature_dict

def serve_as_feature(token_list, feature_serve_as, sentence_list, index,
                     weight_serve_as, feature_dict):
    if token_list[1] == "NNP" or token_list[1] == "NN":
        if (feature_serve_as(sentence_list, index)) is True:
            feature_dict["SERVEASTITLE"] = weight_serve_as
    return feature_dict


def dist_one_token_feature(wnl, sentence_list, weight_before_after_text, index, feature_dict):
    left_right_str = ''
    if index > 0:
        left_right_str += wnl.lemmatize(sentence_list[index - 1][0].lower()) + '-'
    else:
        left_right_str += '-'
    if index < len(sentence_list) - 1: left_right_str += \
        wnl.lemmatize(sentence_list[index + 1][0].lower())
    feature_dict[left_right_str] = weight_before_after_text
    return feature_dict

def dist_two_POS_feature(sentence_list, token_list, POS_twoDistanceTag_TITLE_list,
                         weight_surrounding_two_curr_tags, index, feature_dict):
    tags_str = ''
    if index >= 2:
        tags_str += sentence_list[index - 2][1] + "_"
    else:
        tags_str += "_"
    if index >= 1:
        tags_str += sentence_list[index - 1][1] + "_"
    else:
        tags_str += "_"
    tags_str += token_list[1]
    if index < (len(sentence_list) - 1):
        tags_str += "_" + sentence_list[index + 1][1]
    else:
        tags_str += "_"
    if index < (len(sentence_list) - 2):
        tags_str += "_" + sentence_list[index + 2][1]
    else:
        tags_str += "_"
    if tags_str in POS_twoDistanceTag_TITLE_list:
        feature_dict[tags_str] = weight_surrounding_two_curr_tags
    return feature_dict







"""
10-cross validation / F1 score for evaluation
"""
########## 10-fold cross validation for the logistic regression model ##########
def cross_validation_testing( X, Y, kf, debug=False, param_c=1 ):
    count = 1
    train_score_total = 0
    test_score_total = 0
    for train, test in kf:
        train_X = X[train]
        test_X = X[test]
        train_y = Y[train]
        test_y = Y[test]

        train_score, test_score = _get_F1_score(train_X, train_y,
                                                test_X, test_y, param_c, count, debug)

        train_score_total += train_score
        test_score_total += test_score
        count += 1

    print('========================')
    print("C = "+str(param_c))
    print('avg. training score:\t{}'.format(train_score_total/10))
    print('avg. testing score:\t{}'.format(test_score_total/10))
    print('========================')
    print('\n')
    return (train_score_total/10, test_score_total/10)

def _get_F1_score(train_X, train_y, test_X, test_y, param_c, count, debug_flag=False):
    log_reg = linear_model.LogisticRegression(penalty="l1", C=3)
    log_reg.fit(train_X, train_y)
    pred_y = log_reg.predict(train_X)

    Y_true = []
    Y_pred = []
    for i in range(len(pred_y)):
        if pred_y[i] == "TITLE":
            Y_pred.append(1)
        elif pred_y[i] == "O":
            Y_pred.append(0)
    for i in range(len(train_y)):
        if train_y[i] == "TITLE":
            Y_true.append(1)
        elif train_y[i] == "O":
            Y_true.append(0)
    train_F1_score = f1_score(Y_true, Y_pred)


    pred_y = log_reg.predict(test_X)
    Y_true = []
    Y_pred = []
    for i in range(len(pred_y)):
        if pred_y[i] == "TITLE":
            Y_pred.append( 1 )
        elif pred_y[i] == "O":
            Y_pred.append( 0 )
    for i in range(len(test_y)):
        if test_y[i] == "TITLE":
            Y_true.append( 1 )
        elif test_y[i] == "O":
            Y_true.append( 0 )
    test_F1_score = f1_score(Y_true, Y_pred)


    if debug_flag:
        print( "======= SPLIT "+str(count)+" =======" )
        print( 'training score:\t{}'.format( train_F1_score ) )
        print( 'testing error:\t{}'.format( test_F1_score ) )
        print( '\n' )
    return train_F1_score, test_F1_score

def _get_score(train_X, train_y, test_X, test_y, count, debug_flag=False):
    log_reg = linear_model.LogisticRegression()
    log_reg.fit(train_X, train_y)
    train_score = log_reg.score(train_X, train_y)
    test_score = log_reg.score(test_X, test_y)
    if debug_flag:
        print("======= SPLIT "+str(count)+" =======")
        print('training score:\t{}'.format(train_score))
        print('testing score:\t{}'.format(test_score))
        print('\n')
    return train_score, test_score
########## 10-fold cross validation for the logistic regression model ##########

######################### Calculate F1 score #########################
def calculate_F1_score( testing_set, final_result ):
    Y_true = []
    Y_pred = []

    TP = 0
    FN = 0
    FP = 0
    for i in range(len(testing_set)):
        for j in range(len(testing_set[i])):

            if testing_set[i][j][2] == "TITLE" and final_result[i][j][1] == "TITLE":
                TP += 1
            elif testing_set[i][j][2] == "O" and final_result[i][j][1] == "TITLE":
                FP += 1
                # print("False Positive: " + testing_set[i][j][0])
                # print("Predict: " + "TITLE" + "   Truth: " + "O")
            elif testing_set[i][j][2] == "TITLE" and final_result[i][j][1] == "O":
                FN += 1
                # print("False Negative: " + testing_set[i][j][0])
                # print("Predict: " + "O" + "   Truth: " + "TITLE")
            if testing_set[i][j][2] == "TITLE": Y_true.append(1)
            else: Y_true.append(0)

            if final_result[i][j][1] == "TITLE": Y_pred.append(1)
            else: Y_pred.append(0)

    recall = TP / (TP+FN)
    precise = TP / (TP+FP)
    F1_score_manual = 2*recall*precise / (recall+precise)
    print('------ F1-score manually------')
    print(F1_score_manual)
    print( "Precise: "+str(precise) + "   Recall: "+str(recall) + "\n" )

    print('------ F1-score scikit-learn API ------')
    F1_score = f1_score( Y_true, Y_pred )
    print(F1_score)
######################### Calculate F1 score #########################








"""
Below are bunch of feature engineering methods to find important features to
be trained by logistic regression model, to get high prediction accuracy.
"""

######## Get all Upper case words within training data ########
def feature_engineering_1( training_set ):
    set_for_title = set()
    set_for_not_title = set()

    is_abbrev = 0
    output = open('feature_engineering-for_abbreviation', 'w')
    list_of_TITLE_text = []
    for sentence_list in training_set:
        for i in range(len(sentence_list)):
            token_list = sentence_list[i]

            if token_list[0].lower() in stopwords:
                continue

            token_text = token_list[0]
            pattern = re.compile(r'^[A-Z][a-z]*\.')
            if pattern.match(token_text):
                line = ""
                line += token_text + "    "
                line += "yes" if token_list[2] == "TITLE" else "no"
                line += "\n"

                list_of_TITLE_text.append(line)
    output.writelines( list_of_TITLE_text )
    output.close()

#################### Get around tag for both title and nontitle ####################
def feature_engineering_2( training_set ):
    dict_title_around_tags = defaultdict(int)  # { 'NNP-NP': 20 }
    dict_nontitle_around_tags = defaultdict(int)

    for sentence_list in training_set:
        for i in range(len(sentence_list)):
            token_list = sentence_list[i]
            tags_str = ''

            if i > 0: tags_str += sentence_list[i-1][1] + '__'
            else: tags_str += '__'

            if i < len(sentence_list)-1: tags_str += sentence_list[i+1][1]

            if token_list[2] == 'TITLE':
                dict_title_around_tags[tags_str] += 1
            else:
                dict_nontitle_around_tags[tags_str] += 1

    fp_dict_title = open('trainer__feature_engineering-TITLE_around_tags', 'w')
    dict_title_lines = []
    for tags_str, count in dict_title_around_tags.items():
        line = tags_str + '    ' + str(count) + '\n'
        dict_title_lines.append(line)
    dict_nontitle_lines = []
    for tags_str, count in dict_nontitle_around_tags.items():
        line = tags_str + '    ' + str(count) + '\n'
        dict_nontitle_lines.append(line)

    fp_dict_title = open('trainer__feature_engineering-TITLE_around_tags', 'w')
    fp_dict_title.writelines( dict_title_lines )
    fp_dict_title.close()

    fp_dict_nontitle = open('trainer__feature_engineering-NONTITLE_around_tags', 'w')
    fp_dict_nontitle.writelines(dict_nontitle_lines)
    fp_dict_nontitle.close()

    lines = []
    for tags_str, count in dict_title_around_tags.items():
        line = tags_str + '-----' + 'T: ' + str(count)
        if tags_str in dict_nontitle_around_tags:
            line += "  |   N: " + str(dict_nontitle_around_tags[tags_str])
        else:
            line += "  |"
        line += '\n'
        lines.append( line )
    for tags_str, count in dict_nontitle_around_tags.items():
        if tags_str not in dict_title_around_tags:
            line = tags_str + "-----         |  N: " + str(dict_nontitle_around_tags[tags_str])
            line += '\n'
            lines.append( line )

    fp_dict_nontitle = open('trainer__feature_engineering-NONTITLE_TITLE_around_tokens', 'w')
    fp_dict_nontitle.writelines(lines)
    fp_dict_nontitle.close()

######## Try to find become as a feature ########
def feature_engineering_3( training_set ):
    positive = 0
    negative = 0

    pattern = re.compile(r'\bbecame|become\b', re.IGNORECASE)

    # output = open('feature_engineering-for_or', 'w')
    list_of_TITLE_text = []
    for sentence_list in training_set:
        for i in range(len(sentence_list)):
            token_list = sentence_list[i]

            if token_list[1] == "NN" or token_list[1] == "NNP":
                if i>=1 and pattern.match(sentence_list[i-1][0]):
                    if token_list[2] == "TITLE": positive += 1
                    else: negative += 1
                if i>=2 and pattern.match(sentence_list[i-2][0]):
                    if sentence_list[i-1][1] == "DT":
                        if token_list[2] == "TITLE": positive += 1
                        else: negative += 1
                    if sentence_list[i-1][1] == "JJ":
                        if token_list[2] == "TITLE": positive += 1
                        else: negative += 1
                        if sentence_list[i-1][2] == "TITLE": positive += 1
                        else: negative += 1
                # if i>=3 and pattern.match(sentence_list[i-3][0]):
                #     if sentence_list[i-2][1]=="DT" and sentence_list[i-1][1]=="JJ":
                #         if token_list[2] == "TITLE": positive += 1
                #         else: negative += 1
                #         if sentence_list[i-1][2] == "TITLE": positive += 1
                #         else: negative += 1

    # output.writelines( list_of_TITLE_text )
    # output.close()
    print('become/became+[NN/NNP]:')
    print("P:"+str(positive) + "N:"+str(negative)) # P:46 N:8  good
    print('become/became+DT+[NN/NNP]:')
    print("P:" + str(positive) + "N:" + str(negative)) # P:18 N:9 good
    print('become/became+JJ+[NN/NNP]:')
    print("P:" + str(positive) + "N:" + str(negative)) # P:18 N:2 good
    print('become/became+DT+JJ+[NN/NNP]:')
    print("P:" + str(positive) + "N:" + str(negative)) # P:11 N:25 not good

######## Try to use before two words as the feature ########
def feature_engineering_4( training_set ):
    positive = 0
    negative = 0
    Big_dict = defaultdict(lambda: defaultdict(int))

    output = open('trainer_feature_engineering-BEFORETWOWORDS', 'w')
    lines = []
    for sentence_list in training_set:
        for i in range(len(sentence_list)):
            token_list = sentence_list[i]
            context = ""
            if i >= 2: context += sentence_list[i-2][0].lower() + "_"
            if i >= 1: context += sentence_list[i-1][1].lower()
            # context += token_list[0]

            if token_list[2] == "TITLE": Big_dict[context]["TITLE"] += 1
            else: Big_dict[context]["O"] += 1

    for context in Big_dict:
        line = context + ": "
        line += "YES: " + str(Big_dict[context]["TITLE"]) + "  |  "
        line += "NO: " + str(Big_dict[context]["O"]) + "\n"
        lines.append( line )
    output.writelines( lines )
    output.close()

######## Try to find "as TITLE of" ########
def feature_engineering_5( training_set ):
    output = open('trainer__feature_engineering-as_TITLE_of', 'w')
    list_of_TITLE_text = []

    for sentence_list in training_set:
        for i in range(len(sentence_list)):
            token_list = sentence_list[i]
            if token_list[1] == "NNP" or token_list[1] == "NN":
                (flag, sent_list, j, k) = F_as_TITLE_of(sentence_list, i)
                if flag is True:
                    line = ""
                    sub_text = ""
                    sub_POS = ""
                    for item in sent_list:
                        if item[2] == "TITLE":
                            sub_text += "["+item[0]+"]" + " "
                            sub_POS += "["+item[1]+"]" + " "
                        else:
                            sub_text += item[0] + " "
                            sub_POS += item[1] + " "
                    line += sub_text + "\n" + sub_POS + "\n"
                    line += " ".join( [ item[0] for item in sent_list[j:k] ] ) + "     "
                    line += "YES" if token_list[2] == "TITLE" else "NO"
                    line += "\n\n\n"

                    list_of_TITLE_text.append( line )

    output.writelines(list_of_TITLE_text)
    output.close()

######## Try to find "become [DT/JJ]? TITLE" ########
def feature_engineering_6(training_set):
    output = open('trainer__feature_engineering-become_TITLE', 'w')
    list_of_TITLE_text = []
    pattern_become = re.compile(r'\bbecame|become\b', re.IGNORECASE)

    num_of_title = 0
    num_of_non_title = 0

    for sentence_list in training_set:
        for i in range(len(sentence_list)):
            token_list = sentence_list[i]
            flag = False
            # Use the term 'become/became' appear before and combine POS tag as one feature
            if token_list[1] == "NN" or token_list[1] == "NNP":
                index_end = i+1
                index_begin = -1
                if i >= 1 and pattern_become.match(sentence_list[i - 1][0]):
                    flag = True
                    index_begin = i-1
                elif i >= 2 and pattern_become.match(sentence_list[i - 2][0]):
                    if sentence_list[i - 1][1] == "DT" or sentence_list[i - 1][1] == "JJ":
                        index_begin = i - 2
                        flag = True

                if flag is True:
                    line = ""
                    line += " ".join( [ item[0] for item in
                                        sentence_list[index_begin: index_end]] ) + "   "
                    line += "YES" if token_list[2] == "TITLE" else "NO"
                    line += "\n"
                    list_of_TITLE_text.append(line)

                    if token_list[2] == "TITLE": num_of_title += 1
                    else: num_of_non_title += 1
    output.writelines(list_of_TITLE_text)
    output.close()
    # print("num of title")
    # print(num_of_title)
    # print("num of non title")
    # print(num_of_non_title)

#################### Get all Upper case words within training data ####################
def feature_engineering_7( training_set ):
    is_title = 0
    is_not_title = 0

    output = open('Uppercase_tokens', 'w')
    list_of_TITLE_text = []
    for sentence_list in training_set:
        for i in range(len(sentence_list)):
            token_list = sentence_list[i]
            token_text = token_list[0]

            if token_text.lower() in stopwords:
                continue

            pattern = re.compile(r'^.+?or$', re.IGNORECASE)

            # flag_uppercase = True
            # for char in token_text:
            #     if not re.match(r'\.', char) and not char.isupper():
            #         flag_uppercase = False

            if pattern.match(token_text):
                line = token_text
                if token_list[2] == 'TITLE':
                    line += "    YES"
                    is_title += 1
                else:
                    is_not_title += 1

                line += '\n'
                list_of_TITLE_text.append(line)
    output.writelines( list_of_TITLE_text )
    output.close()

    print("Is_title: ", is_title)
    print("Is_not_tile: ", is_not_title)
#################### Get all Upper case words within training data ####################

def feature_engineering_8(training_set):
    output = open('trainer__feature_engineering-tag', 'w')
    list_of_lines = []

    dict_around_tags = defaultdict(lambda: defaultdict(int))
    for sentence_list in training_set:
        for i in range(len(sentence_list)):
            token_list = sentence_list[i]
            token_text = token_list[2]

            tags_str = ''
            if i>=2: tags_str += sentence_list[i-2][1] + "_"
            else: tags_str += "_"

            if i>=1: tags_str += sentence_list[i-1][1] + "_"
            else: tags_str += "_"

            tags_str += token_list[1]

            if i<(len(sentence_list)-1): tags_str += "_"+sentence_list[i+1][1]
            else: tags_str += "_"

            if i<(len(sentence_list)-2): tags_str += "_"+sentence_list[i+2][1]
            else: tags_str += "_"


            if token_list[2] == "TITLE":
                dict_around_tags[ tags_str ]["T"] += 1
            else:
                dict_around_tags[ tags_str ]["N"] += 1

    for tags_str, dict_T_N in dict_around_tags.items():
        line = ""
        line += tags_str + "    "
        line += "T:" + str(dict_T_N["T"])
        line += " | "
        line += "N:" + str(dict_T_N["N"])
        line += "\n"
        list_of_lines.append( line )
    output.writelines(list_of_lines)
    output.close()
