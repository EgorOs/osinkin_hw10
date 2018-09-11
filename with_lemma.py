#!/usr/bin/env python3

import sklearn.datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import spacy

# Load data

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# random_state guarantees the same pseudo random subset for each run of the program
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)


class LemmaTokenizer(object):
    """
    Found at:
    http://scikit-learn.org/stable/modules/feature_extraction.html
    """
    def __init__(self):
        self.nlp = spacy.load('en', disable=['parser', 'tagger'])
    def __call__(self, doc):
        clean_doc = doc#'\n'.join(doc.split('lines:')[1::]).replace('=',' ').replace('<',' ').replace('>',' ').replace('/',' ').replace('|',' ').replace('\\',' ').replace('@','')
        if len(clean_doc) < 200:
            clean_doc = doc
        sp_data = self.nlp(clean_doc)
        return [token.lemma_ for token in sp_data]


# Create classifier via pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer())),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
        alpha=1e-3, random_state=42,
        max_iter=5, tol=None)),
])


# Evaluate performance
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data

text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))