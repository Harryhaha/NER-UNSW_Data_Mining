# NER-UNSW_Data_Mining
Name Entity Recognition using python and scikit-learn (using logistic regression model)  
Need to do feature engineering to find good features to build an effective classifier.  
  
The entity need to be recognised is TITLE, define a TITLE as an appellation associated with a person by virtue of occupation, office, birth, or as an honorific. For example, in the sentence: "Prime Minister Malcolm Turnbull MP visited UNSW yesterday.", both Prime Minister and MP are TITLEs.  
  
The training data format is like: [ [(’Prime’, ’NNP’), (’Minister’, ’NNP’), (’Malcolm’, ’NNP’),(’Turnbull’, ’NNP’), (’MP’, ’NNP’), (’visit’, ’NN’), (’UNSW’, ’NNP’),(’yesterday’, ’NN’), (’.’, ’.’)], [......], [......] ......... ]  
  
utils.py: All helper functions (weights, feature engineering, feature extraction, evaluation, ......)  
trainer.py: train the model using logistic regression  
tester.py: predict the new data using the trained model  
3 txt files: Gazatteer (part of feature engineering)  
.dat files: Serialised data (for dictVector, model, ..)  




