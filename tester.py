__author__ = "Harry"

from utils import *

path_test_data = sys.argv[1]
path_classifier = sys.argv[2]
path_result = sys.argv[3]

wnl = WordNetLemmatizer()
stopwords = stopwords.words('english')

# Load the dumped dictVector built by trainer
with open("DictVect", 'rb') as f:
    dictVectorizer = pickle.load(f)

# Load the classifier trained by trainer
with open(path_classifier, 'rb') as f:
    classifier = pickle.load(f)

# Load the testing data
with open(path_test_data, 'rb') as f:
    testing_set = pickle.load(f)



"""
Core part: prediction on new data
"""

final_result = []
for sentence_list in testing_set:
    sentence_result = []
    for i in range(len(sentence_list)):
        token_list = sentence_list[i]
        token = token_list[0]
        orig_text = token_list[0]

        features = defaultdict(int)

        # Use the term 'become/became' appear near before and combine POS tag as one feature
        features = BECOME_feature(token_list, sentence_list, W_become, i, features)

        # Below transfer all tokens to lowercase for further processing
        token = wnl.lemmatize(token.lower())

        # Use the combination of the token itself and its POS tag as one feature
        features = token_POS_combine_feature(token, token_list, features)

        # Use Gazatteer as a feature
        features = gazatteer_feature(token, external_knowledge, stopwords_knowledge, features)

        # Use the non-NN/NNP/NNS/NNPS POS of the token as one feature
        features = not_noun_feature(token_list, W_not_noun, features)

        # Use "as [DT]? [NNP/NN] (CC [DT]? [NNP/NN])* of" as one feature
        features = as_of_feature(token_list, F_as_TITLE_of, sentence_list, i,
                                 W_as_title_of, features)

        # Use "serve[s]? as [PRP$/DT/JJ]? ([NN/NNP])+" as one feature
        features = serve_as_feature(token_list, F_serve_as, sentence_list, i, W_serve_as, features)

        # Use each term's left term and right term concatenation as one feature
        features = dist_one_token_feature(wnl, sentence_list, W_before_after_text, i, features)

        # Use centain types of tag combinations of the surrounding 2 tags(including itself tag)
        # as one feature
        features = dist_two_POS_feature(sentence_list, token_list, POS_twoDistanceTag_TITLE_list,
                                            W_surrounding_two_curr_tags, i, features)


        """
        complete the feature extraction and begin to predict each instance using the trained model
        """
        feature_list = dictVectorizer.transform( features )
        _predict_result = classifier.predict( feature_list )
        predict_result = _predict_result[0]
        sentence_result.append( (orig_text, predict_result) )
    final_result.append( sentence_result )


# Dump the final result
with open(path_result, 'wb') as f:
    pickle.dump(final_result, f)
print('testing is end!')

################################ Finish the testing #################################


#################################### Evaluation #####################################
if __name__ == '__main__':
    calculate_F1_score( testing_set, final_result )




