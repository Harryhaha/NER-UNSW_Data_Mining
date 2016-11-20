__author__ = "Harry"

from utils import *

path_train_data = sys.argv[1]
path_classifier = sys.argv[2]


#################### Load and deserialize pickle data ####################
with open(path_train_data, 'rb') as f:
    training_set = pickle.load(f)
#################### Load and deserialize pickle data ####################




"""
Core part: train the model
"""

features_list = []
classifier_list = []

for sentence in training_set:
    for i in range(len(sentence)):
        token_list = sentence[i]
        token = token_list[0]

        features = defaultdict(int)

        # Use the term 'become/became' appear near before and combine POS tag as one feature
        features = BECOME_feature(token_list, sentence, W_become, i, features)

        # Below transfer all tokens to lowercase for further processing
        token = wnl.lemmatize(token.lower())

        # Use the combination of the token itself and its POS tag as one feature
        features = token_POS_combine_feature(token, token_list, features)

        # Use Gazatteer as a feature
        features = gazatteer_feature(token, external_knowledge, stopwords_knowledge, features)

        # Use the non-NN/NNP/NNS/NNPS POS of the token as one feature
        features = not_noun_feature(token_list, W_not_noun, features)

        # Use "as [DT]? [NNP/NN] (CC [DT]? [NNP/NN])* of" as one feature
        features = as_of_feature(token_list, F_as_TITLE_of, sentence, i, W_as_title_of, features)

        # Use "serve[s]? as [PRP$/DT/JJ]? ([NN/NNP])+" as one feature
        features = serve_as_feature(token_list, F_serve_as, sentence, i, W_serve_as, features)

        # Use each term's left term and right term concatenation as one feature
        features = dist_one_token_feature(wnl, sentence, W_before_after_text, i, features)

        # Use centain types of tag combinations of the surrounding 2 tags(including itself tag)
        # as one feature
        features = dist_two_POS_feature(sentence, token_list, POS_twoDistanceTag_TITLE_list,
                             W_surrounding_two_curr_tags, i, features)

        """
        complete the feature extraction and begin to train the model
        """
        features_list.append( features )
        classifier_list.append( token_list[2] )   # Pass in "TITLE" or "O"


dictVectorizer = DictVectorizer()
X = dictVectorizer.fit_transform( features_list )
Y = np.array( classifier_list )

log_reg = linear_model.LogisticRegression( penalty="l1", C=3 )
log_reg.fit(X, Y)

with open(path_classifier, 'wb') as f:
    pickle.dump(log_reg, f)
with open("DictVect", 'wb') as f:
    pickle.dump(dictVectorizer, f)
print('\ntrainging is end!')

################################ Finish the training #################################






