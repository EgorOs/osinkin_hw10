#!/usr/bin/env python3

import sklearn.datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np


# Load data

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# random_state guarantees the same pseudo random subset for each run of the program
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# print(twenty_train.data[0])


# Extract features

# Create bag of words
count_vect = CountVectorizer()
# Create sparse matrix
X_train_counts = count_vect.fit_transform(twenty_train.data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# Training classifier

text_clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)


# Evaluation of performance for naive bayes

twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
X_new_counts = count_vect.transform(docs_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = text_clf.predict(X_new_tfidf)
print(np.mean(predicted == twenty_test.target))


# Support Vector Machine

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
        alpha=1e-3, random_state=42,
        max_iter=5, tol=None)),
])

text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))


# Parameter tuning using grid search

from sklearn.model_selection import GridSearchCV
# Run an exhaustive search of the best parameters on a grid of possible values
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}
# Run parallel calculations on all available cores (n_jobs=-1)
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
twenty_train.target_names[gs_clf.predict(['God is love'])[0]]
print(gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


# Text processing

import spacy
# if doesn't work, run:
# python3 -m spacy download en
# or
# sudo python -m spacy download en
nlp = spacy.load('en')


# Sample text
text = """
The 20 Newsgroups data set is a collection of approximately 20,000
newsgroup documents, partitioned (nearly) evenly across 20 different
newsgroups. To the best of our knowledge, it was originally collected
by Ken Lang, probably for his paper “Newsweeder: Learning to filter
netnews,” though he does not explicitly mention this collection. 
The 20 newsgroups collection has become a popular data set for 
experiments in text applications of machine learning techniques, 
such as text classification and text clustering.
"""

doc = nlp(text)

# for sent in doc.sents:
#     # Print sentences
#     print(sent)

# for token in next(doc.sents):
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#           token.shape_, token.is_alpha, token.is_stop)
#     # print(token.text, token.lemma_)

from io import StringIO

class RenderTokens:
    def __init__(self, sent):
        self.sent = sent
    def _repr_html_(self):
        builder = StringIO()
        builder.write('<table>')
        for token in self.sent:
            builder.write('<tr><td>')
            builder.write(
                '</td><td>'.join(
                    (token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                     token.shape_, str(token.is_alpha), str(token.is_stop))
                )
            )
            builder.write('</td></tr>')
        builder.write('</table>')
        return builder.getvalue()

f = open('output.htm', 'w')
f.write(RenderTokens(next(doc.sents))._repr_html_())
f.close()