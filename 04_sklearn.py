
### Learning Resources ###

# 1. scikit-learn User Guide
# https://scikit-learn.org/stable/user_guide.html
# 2. scikit-learn API Reference
# https://scikit-learn.org/stable/modules/classes.html
# 3. Alexander Ihler from UC Irvine
# https://www.youtube.com/watch?v=qPhMX0vb6D8&list=PL_Ig1a5kxu56fDjM0hshzhauzT9MWpO7K
# 4. Andrew's Machine Learning course on Coursera
# https://www.coursera.org/learn/machine-learning
# https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN
# https://www.youtube.com/watch?v=6QRpDLj8huE&list=PLoR5VjrKytrCv-Vxnhp5UyS1UjZsXP0Kj

# Important scikit-learn User Guide Topics
# 1. Supervised Learning
#   a. Linear Models
#       - linear regression
#       - logistic regression
#   b. Ensemble Methods
#       - random forests
#       - gradient boosting
# 2. Unsupervised Learning
#   a. Clustering (k-means)
# 3. Model Selection and Evaluation
#   a. Cross-validation
#   b. Metrics and scoring (model performance evaluation)
#   c. Tuning the hyper-parameters (important: regularization)
#   d. Validation curves
# 4. Dataset Transformations
#   a. Preprocessing data
#   b. Transforming the prediction target
#   c. Feature extraction
#   d. Imputation of missing values
#   e. Pairwise metrics (i.e., similarity or distance)
#   f. Pipelines and composite estimators
#   g. Unsupervised dimensionality reduction


##############################
### Part 1. Basic Concepts ###
##############################

### 1.1 What is Machine Learning?

## Terminology
# Data Science (DS)
# Analytics
# Statistics
# Artificial Intelligence (AI)
# Machine Learning (ML)
# Neural Networks (NN)
# Deep Learning (DL) - a.k.a. Deep Neural Networks

# https://www.ibm.com/cloud/blog/ai-vs-machine-learning-vs-deep-learning-vs-neural-networks
# https://ai.plainenglish.io/artificial-intelligence-vs-machine-learning-vs-deep-learning-whats-the-difference-dccce18efe7f

## Paradigm Shift in Thinking
# machine learning vs. rule-based coding (a.k.a. hard coding)
# stochastic vs. deterministic
# statistics vs. calculus
# govern with intervention without interference


### 1.2 Supervised Learning
# regression vs. classification - mapping X to Y via models
# linear regression vs. logistic regression
# model = algorithm + parameters - model specification
# hyper-parameters pertain to
# (a) data preparation, e.g. features inclusion, features engineering
# (b) model specification, e.g. algorithm choice, model architecture
# (c) model training, e.g. validation scheme, optimizer, batch-size, epoch

#               train & improve             evaluate
# humans        hyper-parameter tuning      performance metrics
# computers     gradient descent            loss function
# Note: gradient descent is automatic parameter tuning

# model training vs. model improvement
# parameters vs. hyper-parameters
# loss function vs. performance metrics
# gradient descent vs. hyper-parameter tuning


### 1.3 Bias-Variance Tradeoff

## Balance is Everything
# under-fitting vs. over-fitting
# bias vs. variance
# https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76
# https://www.machinelearningtutorial.net/2017/01/26/the-bias-variance-tradeoff/

#                   under-fit   over-fit
# complex model     ---         +++
# validation                    ---
# regularization    +++         ---
# more features     ---         +++
# more data                     ---
# early stopping                ---
# ensemble                      ---
# dropouts                      ---
# data augmentation             ---
# https://elitedatascience.com/overfitting-in-machine-learning
# https://medium.com/analytics-vidhya/7-ways-to-avoid-overfitting-9ff0e03554d3
# https://towardsdatascience.com/8-simple-techniques-to-prevent-overfitting-4d443da2ef7d

## Validation Schemes
# training-validation-test scheme
# cross validation scheme

## Regularization
# l1-norm regularization - lasso regression
# l2-norm regularization - ridge regression
# draw a validation curve to make the regularization decision
# https://www.youtube.com/watch?v=sO4ZirJh9ds
# https://www.youtube.com/watch?v=Xm2C_gTAl8c
# https://www.kaggle.com/residentmario/l1-norms-versus-l2-norms
# https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261

## Early Stopping
# draw a validation curve to make the early-stopping decision

## Learning Curves
# https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html

## Validation vs. Learning Curves
#               validation curves               learning curves
# decisions     hyper-parameter tuning          under-fitting vs. over-fitting
# X-axis        range of a hyper-parameter      number of training samples
# Y-axis        loss or performance metrics     loss or performance metrics


### 1.4 Performance Metrics

## regression
# - negative of mean squared error (MSE)
# - negative of root of mean squared error (RMSE)

## classification
# binary classification
# - accuracy
# - precision
# - recall
# - f1 score
# multi-class classification
# multi-label classification
# - micro-average {...}
# - macro-average {...}
# - weighted-average {...}
# https://www.kaggle.com/enforcer007/what-is-micro-averaged-f1-score
# https://stackoverflow.com/questions/55740220/macro-vs-micro-vs-weighted-vs-samples-f1-score



########################################
### Part 2. Housing Price Prediction ###
########################################

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline

np.set_printoptions(suppress=True)      # suppress scientific notation


### 2.1 Load Data

boston = load_boston()
print(boston.DESCR)
print(boston.feature_names)

X = boston.data
Y = boston.target
print(X.shape, Y.shape)

X_tran, X_test, y_tran, y_test = train_test_split(X, Y, shuffle=True, test_size=0.1, random_state=2021)
print(X_tran.shape, X_test.shape)
print(y_tran.shape, y_test.shape)

X_trn, X_val, y_trn, y_val = train_test_split(X_tran, y_tran, shuffle=True, test_size=0.1, random_state=2021)
print(X_trn.shape, X_val.shape)
print(y_trn.shape, y_val.shape)

### 2.2 Train Regression Models with Regularization

## train a vanilla linear regression
regressor = LinearRegression()                  # algorithm
regressor = regressor.fit(X_tran, y_tran)       # model > Model | fit == train
y_pred = regressor.predict(X_test)
y_pred[:5]
y_test[:5]

r2 = r2_score(y_test, y_pred)
print(r2)

mse = mean_squared_error(y_test, y_pred)
print(mse, np.sqrt(mse))
print(y_test.mean())

## write a function to train various models
def fit_linear(X_tran, y_tran, X_test, y_test, model='linear', metric='r2', **kwargs):
    if model == 'linear':
        regressor = LinearRegression()
    elif model == 'lasso':
        regressor = Lasso(**kwargs)
    elif model == 'ridge':
        regressor = Ridge(**kwargs)
    elif model == 'elastic':
        regressor = ElasticNet(**kwargs)
    else:
        Exception("Unknown model: {}".format(model))

    regressor.fit(X_tran, y_tran)
    y_pred = regressor.predict(X_test)
    if metric == 'r2':
        r2 = r2_score(y_test, y_pred)
        print("R-Square is {}".format(r2))
    else:
        mse = mean_squared_error(y_test, y_pred)
        print("MSE is {}, RMSE is {}".format(round(mse, 4), round(np.sqrt(mse), 4)))
    return regressor


model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'linear')

model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'lasso', alpha=1.0)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'lasso', alpha=3.0)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'lasso', alpha=0.03)

model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'ridge', alpha=1.0)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'ridge', alpha=3.0)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'ridge', alpha=0.3)

model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'elastic', alpha=1.0, l1_ratio=0.5)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'elastic', alpha=3.0, l1_ratio=0.5)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'elastic', alpha=0.3, l1_ratio=0.5)


## normalize all features
X_tran, X_test, y_tran, y_test = train_test_split(X, Y, shuffle=True, test_size=0.1, random_state=2021)
print(X_tran.mean(0))
print(X_tran.std(0))
print(X_test.mean(0))
print(X_test.std(0))

scaler = StandardScaler()
scaler = scaler.fit(X_tran)
X_tran = scaler.transform(X_tran)
X_test = scaler.transform(X_test)
print(X_tran.mean(0))
print(X_tran.std(0))
print(X_test.mean(0))
print(X_test.std(0))

model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'linear')

model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'lasso', alpha=1.0)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'lasso', alpha=3.0)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'lasso', alpha=0.3)

model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'ridge', alpha=1.0)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'ridge', alpha=3.0)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'ridge', alpha=0.3)

model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'elastic', alpha=1.0, l1_ratio=0.5)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'elastic', alpha=3.0, l1_ratio=0.5)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'elastic', alpha=0.3, l1_ratio=0.5)


### 2.3 Tune Hyper-parameters with Grid Search
X_tran, X_test, y_tran, y_test = train_test_split(X, Y, shuffle=True, test_size=0.1, random_state=2021)
print(X_tran.shape, X_test.shape)

scaler = StandardScaler()
X_tran = scaler.fit_transform(X_tran)
X_test = scaler.transform(X_test)

regressor = ElasticNet()
parameters = {
    'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}
gs = GridSearchCV(
    estimator=regressor,
    param_grid=parameters,
    cv=10,
    scoring='r2',
    # scoring='neg_mean_squared_error',
    n_jobs=-1,
    return_train_score=True
)

gs = gs.fit(X_tran, y_tran)
y_pred = gs.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)
mse = mean_squared_error(y_test, y_pred)
print(mse, np.sqrt(mse))

cvdat = pd.DataFrame(gs.cv_results_)
cvdat.iloc[0, :]
cvdat = cvdat.sort_values(by='rank_test_score')
cvdat.iloc[0, :]
cvdat['params']

# can specify cross-validation scheme & scoring method
cv = KFold(n_splits=10, shuffle=False, random_state=None)
cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=None)

r2 = make_scorer(r2_score)
mse = make_scorer(mean_squared_error, greater_is_better=False)

gs = GridSearchCV(
    estimator=regressor,
    param_grid=parameters,
    cv=cv,
    scoring=r2,
    # scoring=mse,
    n_jobs=-1,
    return_train_score=True
)

gs = gs.fit(X_tran, y_tran)
y_pred = gs.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)
mse = mean_squared_error(y_test, y_pred)
print(mse, np.sqrt(mse))

cvdat = pd.DataFrame(gs.cv_results_)
cvdat = cvdat.sort_values(by='rank_test_score')
cvdat.iloc[0]


### 2.4 Perform Feature Engineering
poly = PolynomialFeatures(degree=2)
mat = np.arange(6).reshape(3, 2); mat
poly.fit_transform(mat)

X_tran, X_test, y_tran, y_test = train_test_split(X, Y, shuffle=True, test_size=0.1, random_state=2021)
print(X_tran.shape, X_test.shape)

poly = PolynomialFeatures(degree=3, include_bias=False)
X_tran = poly.fit_transform(X_tran)
X_test = poly.transform(X_test)
print(X_tran.shape, X_test.shape)

scaler = StandardScaler()
X_tran = scaler.fit_transform(X_tran)
X_test = scaler.transform(X_test)

model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'linear')

model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'lasso', alpha=1.0)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'lasso', alpha=3.0)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'lasso', alpha=0.3)

model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'ridge', alpha=1.0)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'ridge', alpha=3.0)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'ridge', alpha=0.3)

model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'elastic', alpha=1.0, l1_ratio=0.5)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'elastic', alpha=3.0, l1_ratio=0.5)
model_linear = fit_linear(X_tran, y_tran, X_test, y_test, 'elastic', alpha=0.3, l1_ratio=0.5)


regressor = Ridge()
parameters = {
    'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
}
gs = GridSearchCV(
    estimator=regressor,
    param_grid=parameters,
    cv=10,
    scoring='r2',
    # scoring='neg_mean_squared_error',
    n_jobs=-1,
    return_train_score=True
)

gs = gs.fit(X_tran, y_tran)
y_pred = gs.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)
mse = mean_squared_error(y_test, y_pred)
print(mse, np.sqrt(mse))

cvdat = pd.DataFrame(gs.cv_results_)
cvdat = cvdat.sort_values(by='rank_test_score')
cvdat.iloc[0, :]
cvdat[['params', 'mean_test_score']]
cvdat.loc[0, 'params']


### 2.5 Create a Grid-Searchable Pipeline

## create a pipeline
X_tran, X_test, y_tran, y_test = train_test_split(X, Y, shuffle=True, test_size=0.1, random_state=2021)
print(X_tran.shape, X_test.shape)

pipe = Pipeline([
    ('polynomial', PolynomialFeatures(include_bias=False)),     # must have fit & transform methods
    ('scaler', StandardScaler()),                               # must have fit & transform methods
    ('regressor', Ridge(alpha=0.3))                             # must be an estimator (i.e. regressor or classifier)
])

pipe = pipe.fit(X_tran, y_tran)
y_pred = pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)
mse = mean_squared_error(y_test, y_pred)
print(mse)

## combine pipeline with gridsearch
pipe = Pipeline([
    ('polynomial', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Ridge())
])
parameters = {
    'polynomial__degree': [2, 3],
    'regressor__alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
}
gs = GridSearchCV(
    estimator=pipe,
    param_grid=parameters,
    cv=10,
    scoring='r2',
    # scoring='neg_mean_squared_error',
    n_jobs=-1,
    return_train_score=True
)

gs = gs.fit(X_tran, y_tran)
y_pred = gs.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)
mse = mean_squared_error(y_test, y_pred)
print(mse)

cvdat = pd.DataFrame(gs.cv_results_)
cvdat = cvdat.sort_values(by='rank_test_score')
cvdat.iloc[0, :]
cvdat[['params', 'mean_test_score']]
_ = [print(x) for x in cvdat['params'].iloc[0:5]]


### Question: Try Regression with New Data
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing(as_frame=True)
print(california.DESCR)
X = california.data
Y = california.target
print(X.shape, Y.shape)
print(california.feature_names)


###################################
### Part 3. Text Classification ###
###################################
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer    # Features Extraction
from sklearn.model_selection import train_test_split                            # Data Splitter
from sklearn.linear_model import LogisticRegression                             # Classifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier       # Classifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier                                               # Classifier
from catboost import CatBoostClassifier                                         # Classifier
from lightgbm import LGBMClassifier                                             # Classifier
from sklearn.model_selection import RandomizedSearchCV                          # Hyper-parameter Tuning
from sklearn.pipeline import Pipeline                                           # Pipeline
from sklearn.metrics import accuracy_score, classification_report               # Performance Metrics
from sklearn.metrics import precision_score, recall_score, f1_score             # Performance Metrics
np.set_printoptions(suppress=True)


### 3.1 Text Preprocessing/Cleaning
# ***   1. lower strings            I Like It, I LIKE IT -> i like it
#       2. handle contractions      he's -> hes, he's -> he is, he has
#       3. handle abbreviations     u.s.a. -> usa, u.s.a. -> united states of american
# **    4. handle numbers           134 -> math_number, 2:30 -> datetime_time, July 2014 -> datetime_date
# ***   5. remove punctuations      Note: not including '_' or '-' between words, e.g. 'math_number', 'long-term'
# *     6. lemmatization            apples -> apple, goes, went, going, gone -> go
#          stemming                 trouble, troubling, troubled -> troubl, universe, university, universal -> univ
# **    7. remove stopwords         Note: may keep negations, e.g. 'no', 'not', 'none', 'against', 'never'
#       8. remove spaces            ' ' \t \n


## create stylized examples
import re
import spacy
nlp = spacy.load('en_core_web_sm')
# $ python -m spacy download en_core_web_sm
from nltk.stem import PorterStemmer, LancasterStemmer
stemmer = PorterStemmer()

d1 = "We're having a.b.c.d or i.e. u.s.a and 16.2 and U.K. and others."
d2 = "We buy 12 apples, good apples, from Ms. ABC at 16.2 dollars/pound 24/7 from Monday-Friday. How's that?"
d3 = "1. I won't eat 1/12 of the #1 top cakes. I got 1.2 dollars or 1.2usd and a long-term/short_term goal."
d4 = "I lost 22 pounds at 9:30, 11:05 or 1:30pm or 2pm or 3 pm or 1-2pm on 2016-01-02 or 01-02-2016."
d5 = "  --He's a dedicated person. \t He dedicated his time to work! \n --are you sure?  "
d6 = "I am interested in these interesting things that interested me."
docs = [d1, d2, d3, d4, d5, d6]

## define text preprocessing functions
def lemmatize(text):
    tuples = [(str(token), token.lemma_) for token in nlp(text)]
    text = ' '.join([lemma if lemma != '-PRON-' else token for token, lemma in tuples])
    text = re.sub(' ([_-]) ', '\\1', text)  # 'short - term' -> 'short-term'
    return text

def text_cleaner(text, option='none'):
    stopwords = set(['i', 'I', 'you', 'he', 'she', 'it', 'we', 'they', 'am', 'is', 'are'])
    text = text.lower()                                                     # 1. lower strings
    text = text.replace("'s", 's')                                          # 2. handle contractions
    text = re.sub('([a-zA-Z])\\.([a-zA-Z])\\.*', '\\1\\2', text)            # 3. handle abbreviations
    text = re.sub('\\d+', '', text)                                         # 4. remove numbers
    text = re.sub('[^a-zA-Z0-9\\s_-]', ' ', text)                           # 5. remove punctuations
    if 'lemma' in option:
        text = lemmatize(text)                                              # 6. perform lemmatization
    elif 'stem' in option:
        text = ' '.join([stemmer.stem(x) for x in text.split(' ')])         # 6. perform stemming
    text = ' '.join([x for x in text.split(' ') if x not in stopwords])     # 7. remove stopwords
    text = ' '.join([x.strip('_-') for x in text.split()])                  # 8. remove spaces, '_' and '-'
    return text

## clean texts
option = 'none'
option = 'lemmatization'
option = 'stemming'

clean_docs = [text_cleaner(doc, option) for doc in docs]; clean_docs
for raw, clean in zip(docs, clean_docs):
    print("original doc:", raw)
    print("cleaned  doc:", clean)
    print()


## define a sklearn Transformer class
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin, RegressorMixin
from tqdm import tqdm

class TextCleaner(BaseEstimator):
    d1 = "We're having a.b.c.d or i.e. u.s.a and 16.2 and U.K. and others."
    d2 = "We buy 12 apples, good apples, from Ms. ABC at 16.2 dollars/pound 24/7 from Monday-Friday. How's that?"
    d3 = "1. I won't eat 1/12 of the #1 top cakes. I got 1.2 dollars or 1.2usd and a long-term/short_term goal."
    d4 = "I lost 22 pounds at 9:30, 11:05 or 1:30pm or 2pm or 3 pm or 1-2pm on 2016-01-02 or 01-02-2016."
    d5 = "  --He's a dedicated person. \t He dedicated his time to work! \n --are you sure?  "
    d6 = "I am interested in these interesting things that interested me."
    sample_docs = [d1, d2, d3, d4, d5, d6]

    def __init__(self, option='none'):
        self.option = option
        self.nlp = spacy.load('en_core_web_sm')
        self.stemmer = PorterStemmer()

    def lemmatize(self, text):
        tuples = [(str(token), token.lemma_) for token in self.nlp(text)]
        text = ' '.join([lemma if lemma != '-PRON-' else token for token, lemma in tuples])
        text = re.sub(' ([_-]) ', '\\1', text)  # 'short - term' -> 'short-term'
        return text

    def text_cleaner(self, text):
        stopwords = set(['i', 'I', 'you', 'he', 'she', 'it', 'we', 'they', 'am', 'is', 'are'])
        text = text.lower()                                                     # 1. lower strings
        text = text.replace("'s", 's')                                          # 2. handle contractions
        text = re.sub('([a-zA-Z])\\.([a-zA-Z])\\.*', '\\1\\2', text)            # 3. handle abbreviations
        text = re.sub('\\d+', '', text)                                         # 4. remove numbers
        text = re.sub('[^a-zA-Z0-9\\s_-]', ' ', text)                           # 5. remove punctuations
        if 'lemma' in self.option:
            text = self.lemmatize(text)                                         # 6. perform lemmatization
        elif 'stem' in self.option:
            text = ' '.join([self.stemmer.stem(x) for x in text.split(' ')])    # 6. perform stemming
        text = ' '.join([x for x in text.split(' ') if x not in stopwords])     # 7. remove stopwords
        text = ' '.join([x.strip('_-') for x in text.split()])                  # 8. remove spaces, '_' and '-'
        return text

    def transform(self, docs):
        clean_docs = []
        self.fails = []
        for doc in tqdm(docs):
            try:
                clean_docs.append(self.text_cleaner(doc))
            except:
                self.fails.append(doc)
        if len(self.fails) > 0:
            print("Some documents failed to be converted. Check self.fails for failed documents")
        return clean_docs

    def fit(self, docs, y=None):            # Warning: must return self not None
        return self

    def fit_transform(self, docs, y=None):
        return self.fit(docs, y).transform(docs)


cleaner = TextCleaner(option='lemma')
docs = cleaner.sample_docs

clean_docs = cleaner.transform(docs); clean_docs
for raw, clean in zip(docs, clean_docs):
    print("original doc:", raw)
    print("cleaned  doc:", clean)
    print()


### 3.2 Text Encoding
d1 = 're have abcd or ie usa and and uk and other'
d2 = 'buy apple good apple from ms abc at dollar pound from monday-friday how s that'
d3 = 'win t eat of the top cake I get dollar or usd and a long-term short_term goal'
d4 = 'lose pound at or pm or pm or pm or pm on  or '
d5 = 'hes a dedicated person dedicate his time to work are sure'
d6 = 'be interested in these interesting thing that interest'
docs = [d1, d2, d3, d4, d5, d6]

vec = TfidfVectorizer(
    input='content',                    # default; {'filename', 'file', 'content'}; input must be a list
    encoding='utf-8',                   # default; same options as in str.decode(encoding='utf-8')
    decode_error='strict',              # default; same options as in str.decode(errors='strict')
    # preprocessing arguments
    lowercase=True,                     # default; convert strings to lowercase
    strip_accents='unicode',            # default=None; remove accents; 'unicode' is slower but universal
    preprocessor=None,                  # default
    # tokenization arguments
    analyzer='word',                    # default; {'word', 'char', 'char_wb'}
    token_pattern=u'(?u)\\b\\w\\w+\\b', # default; equivalent to using a nltk.RegexpTokenizer()
    tokenizer=None,                     # default; can use a nltk tokenizer
    # vocabulary arguments
    stop_words=None,                    # default=None; try stop_words='english' for example
    ngram_range=(1, 2),                 # default=(1,1)
    min_df=1,                           # default=1  ; int or [0.0, 1.0]; ignore terms with a doc-freq < cutoff
    max_df=0.8,                         # default=1.0; [0.0, 1.0] or int; ignore terms with a doc-freq > cutoff
    max_features=None,                  # default; keep only the top max_features ordered by term-freq
    vocabulary=None,                    # default; if provided, max_df, min_df, max_features are ignored
    # TF-IDF adjustment arguments
    binary=False,                       # default; if True, all non-zero term counts are set to 1
    sublinear_tf=True,                  # default; if True, use 1 + log(tf) for non-zeros; else use tf
    use_idf=True,                       # default; if True, enable IDF re-weighting
    smooth_idf=True,                    # default; if True, use 1 + log((N_docs+1)/(df+1)), else use 1 + log(N_docs/df)
    norm='l2',                          # default; perform post TF-IDF normalization such that output row has unit norm
)

## Understand TfidfVectorizer - before performing vec.fit_transform()
preprocessor = vec.build_preprocessor()             # lower strings & remove accents
preprocessor("Let's TRY this OUT: aigué buâlaè")

tokenizer = vec.build_tokenizer()                   # tokenize documents
tokenizer("let's try this out: aigue bualae")

analyzer = vec.build_analyzer()                     # preprocess > tokenize > remove stopwords > get n-grams > prune vocabulary
analyzer("Let's TRY this OUT: aigué buâlaè")        # i.e., all steps before TF-IDF adjustment

## Understand TfidfVectorizer - after performing vec.fit_transform()
dtm = vec.fit_transform(docs); dtm                  # scipy sparse csr matrix
idx2term = vec.get_feature_names(); idx2term        # a list mapping indices to terms
term2idx = vec.vocabulary_; term2idx                # a dict mapping terms to indices

vec.get_stop_words()                                # if stop_words argument is provided
vec.stop_words_                                     # a set of terms that are ignored due to max_df, min_df, or max_features

dtm = pd.DataFrame(dtm.todense())                   # convert DTMatrix to DataFrame
dtm.columns = vec.get_feature_names()
dtm


### 3.3 Load Data
from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
remove = ('headers', 'footers', 'quotes')
data = fetch_20newsgroups(subset='all', categories=categories, remove=remove, shuffle=True, random_state=2021)
label2name = {i: x for i, x in enumerate(data.target_names)}; label2name
name2label = {x: i for i, x in enumerate(data.target_names)}; name2label
df = pd.DataFrame({'label_name': data.target, 'label': data.target, 'text': data.data}); df[:5]
df['label_name'] = df['label_name'].map(label2name); df[:5]
print(df.shape)

# Clean Text
option = 'none'  # Speed: 'none' > 'stem' > 'lemma'
cleaner = TextCleaner(option=option)
X = cleaner.transform(df['text'])
Y = df['label'].astype('int16').to_list()
print(len(X), len(Y))

# Split Training & Test Data
X_tran, X_test, y_tran, y_test = train_test_split(X, Y, test_size=0.1, random_state=2021)
print('X_tran shape:', len(X_tran), '\t', 'Y_tran shape:', len(y_tran))
print('X_test shape:', len(X_test), '\t', 'Y_test shape:', len(y_test))

# Create Document-Term Matrix
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    max_features=4000,
    sublinear_tf=True
)
X_tran = vectorizer.fit_transform(X_tran).toarray()
X_test = vectorizer.transform(X_test).toarray()
print('X_tran shape:', X_tran.shape, '\t', 'Y_tran shape:', len(y_tran))
print('X_test shape:', X_test.shape, '\t', 'Y_test shape:', len(y_test))


### 3.4 Logistic Regression
# https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html

## multi-class classification
logreg = LogisticRegression(
    penalty='l2',               # default; {'l1', 'l2', 'elasticnet', 'none'}
    l1_ratio=None,              # range (0, 1) used if penalty='elasticnet'
    C=1.0,                      # default; inverse of alpha - smaller values give stronger regularization
    multi_class='auto',         # default; {'auto', 'ovr', 'multinomial'} handles multi-class only
    solver='liblinear',         # default; {'liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'}
    max_iter=100,               # default;
    tol=1e-4,                   # default; tolerance for stopping criteria
)
# Note 'ovr' fits multiple binary classifiers while 'multinomial' fits one multi-class classifier

logreg.fit(X_tran, y_tran)
y_prob = logreg.predict_proba(X_test)
print(y_prob.shape); y_prob[:5]
y_pred = logreg.predict(X_test)
print(y_pred.shape); y_pred[:5]

print('accuracy:',           accuracy_score(y_test, y_pred).round(4))
print('precision-macro:',    precision_score(y_test, y_pred, average='macro').round(4))
print('precision-micro:',    precision_score(y_test, y_pred, average='micro').round(4))
print('precision-weighted:', precision_score(y_test, y_pred, average='weighted').round(4))
print('recall-macro:',       recall_score(y_test, y_pred, average='macro').round(4))
print('recall-micro:',       recall_score(y_test, y_pred, average='micro').round(4))
print('recall-weighted:',    recall_score(y_test, y_pred, average='weighted').round(4))
print('f1-macro:',           f1_score(y_test, y_pred, average='macro').round(4))
print('f1-micro:',           f1_score(y_test, y_pred, average='micro').round(4))
print('f1-weighted:',        f1_score(y_test, y_pred, average='weighted').round(4))

print(classification_report(y_test, y_pred, digits=4))
pdat = pd.DataFrame(classification_report(y_test, y_pred, digits=4, output_dict=True)).round(4).T; pdat
pdat[:4].mean(0).round(4)

## train with a pipeline
X = df['text'].to_list(); X[:5]
Y = df['label'].astype('int16').to_list(); Y[:5]
print(len(X), len(Y))

X_tran, X_test, y_tran, y_test = train_test_split(X, Y, test_size=0.1, random_state=2021)

pipe = Pipeline([
    ('cleaner', TextCleaner()),                     # must have fit & transform methods
    ('tfidf', TfidfVectorizer()),                   # must have fit & transform methods
    ('clf', LogisticRegression(solver='saga'))      # must be an estimator (i.e. regressor or classifier)
])
parameters = {
    'cleaner__option': ['none'],                    # cleaner options matter much less than 'tfidf' and 'clf' options
    'tfidf__ngram_range': [[1, 2], [1, 3]],
    'tfidf__min_df': [5, 10],
    'tfidf__max_df': [0.8, 0.9, 0.95],
    'tfidf__max_features': [4000, 8000, None],
    'clf__penalty': ['none', 'l1', 'l2'],           # Warning: some solver does not support certain penalty types
    'clf__C': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
}
rs = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=parameters,     # 'param_grid' in GridSearch
    n_iter=2,                           # default=10 - THIS IS NEW
    cv=3,
    scoring='f1_macro',
    n_jobs=1,                           # Warning: using n_jobs=-1 may result in memory failures
    return_train_score=True
)
# Warning: You should test out the most memory-demanding set of parameters first before performing RandomizedSearchCV

rs = rs.fit(X_tran, y_tran)
y_pred = rs.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

cvdat = pd.DataFrame(rs.cv_results_); cvdat
cvdat = cvdat.sort_values(by='rank_test_score')
cvdat.T
cvdat.iloc[0, :]['params']


## multi-label classification
from sklearn.datasets import make_multilabel_classification
X_mlab, Y_mlab = make_multilabel_classification(
    n_samples=500,
    n_features=20,
    n_classes=4,
    n_labels=2,
    allow_unlabeled=False,
    random_state=1
)
X_tran_mlab, X_test_mlab, y_tran, y_test = train_test_split(X_mlab, Y_mlab, test_size=0.1, random_state=2021)
print('X_tran shape:', X_tran_mlab.shape, '\t', 'Y_tran shape:', y_tran.shape)
print('X_test shape:', X_test_mlab.shape, '\t', 'Y_test shape:', y_test.shape)

clf = OneVsRestClassifier(estimator=logreg)
clf.fit(X_tran_mlab, y_tran)
y_prob = clf.predict_proba(X_test_mlab)
print(y_prob.shape); y_prob[:5]
y_pred = clf.predict(X_test_mlab)
print(y_pred.shape); y_pred[:5]
print(classification_report(y_test, y_pred, digits=4))


### 3.5 Random Forest
# https://victorzhou.com/blog/information-gain/
cleaner = TextCleaner(option='none')
X = cleaner.transform(df['text'])
Y = df['label'].astype('int16').to_list()
print(len(X), len(Y))
X_tran, X_test, y_tran, y_test = train_test_split(X, Y, test_size=0.1, random_state=2021)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    max_features=4000,
    sublinear_tf=True
)
X_tran = vectorizer.fit_transform(X_tran).toarray()
X_test = vectorizer.transform(X_test).toarray()
print('X_tran shape:', X_tran.shape, '\t', 'Y_tran shape:', len(y_tran))
print('X_test shape:', X_test.shape, '\t', 'Y_test shape:', len(y_test))

# create a simplest RF classifier
rf = RandomForestClassifier(n_jobs=-1)

# important arguments breakdown (Note: M, N = X.shape)
rf = RandomForestClassifier(    # default   range           prevents overfitting
    n_estimators=100,           # 100       [1, inf)        larger
    criterion='gini',           # gini      {'gini', 'entropy'}
    max_depth=None,             # None      [1, inf)        smaller
    min_samples_split=2,        # 2         [2, M)          larger
    min_samples_leaf=1,         # 1         [1, M)          larger
    max_features='auto',        # auto   [1, N] or (0, 1]   smaller
    max_samples=0.8,            # M      (0, 1) or [1, M]   smaller
    min_impurity_decrease=0.0,  # 0.0       [0, inf)        larger
    class_weight=None,          # None
    n_jobs=-1,                  # 1         [1, -1]
    random_state=None,          # None
    warm_start=False,           # False
    bootstrap=True,             # True
)

rf.fit(X_tran, y_tran)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))


### 3.6 Gradient Boosting

## XGB Classifier
# https://xgboost.readthedocs.io/en/latest/parameter.html
# https://xgboost.readthedocs.io/en/latest/tutorials/index.html
# https://www.kaggle.com/stuarthallows/using-xgboost-with-scikit-learn
X_trn, X_val, y_trn, y_val = train_test_split(X_tran, y_tran, test_size=0.1, random_state=2020)

# Train One Model
xgb = XGBClassifier(        # alias     default     range       prevents overfitting
    objective='multi:softmax',
    n_estimators=100,       # ----      100         [1, inf)    larger
    max_depth=20,           # ----      3           [1, inf)    smaller
    learning_rate=0.3,      # eta       0.1         [0, 1]      smaller
    max_bin=255,            # ----      256         [2, inf)    smaller
    max_leaves=0,           # ----      0           [0, inf)    smaller
    min_split_loss=0.003,   # gamma     0           [0, inf)    larger
    min_child_weight=1,     # ----      1           [0, inf)    larger
    subsample=0.6,          # ----      1           (0, 1]      smaller
    colsample_bytree=0.2,   # ----      1           (0, 1]      smaller
    colsample_bylevel=0.2,  # ----      1           (0, 1]      smaller
    colsample_bynode=0.2,   # ----      1           (0, 1]      smaller
    reg_alpha=0.03,         # l1 alpha  0           [0, inf)    larger
    reg_lambda=0.3,         # l2 lambda 1           [0, inf)    larger
    scale_pos_weight=1,     # ----      1
    seed=0,                 # ----      0
    n_jobs=4,               # ----      1
    verbosity=2             # ----      1           [0,1,2,3]
)
xgb.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], early_stopping_rounds=5, verbose=1)
y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))


# Random Search
param_grid = {'max_depth': [10, 20], 'num_leaves': [20, 30]}
rs = RandomizedSearchCV(xgb, param_grid, n_iter=3, cv=3, n_jobs=-1)
rs.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], early_stopping_rounds=3)
y_pred = rs.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))


## LightGBM Classifier
# https://lightgbm.readthedocs.io/en/latest/Parameters.html
# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
X_trn, X_val, y_trn, y_val = train_test_split(X_tran, y_tran, train_size=0.9, random_state=2020)

# Train One Model
gbm = LGBMClassifier(       # alias                     default     range       prevents overfitting
    task='train',           # ----                      train
    objective='multiclass', # ----                      regression
    n_estimators=500,       # num_iterations            100         [1, inf)    larger
    max_depth=30,           # ----                      -1          [1, inf)    smaller
    learning_rate=0.1,      # eta                       0.1         (0, inf)    smaller
    max_bin=255,            # ----                      255         [2, inf)    smaller
    num_leaves=40,          # max_leaves                31          [2, 131072] smaller
    min_child_samples=2,    # min_data_in_leaf          20          [0, inf)    larger
    min_split_gain=0.003,   # min_gain_to_split         0.0         [0, inf)    larger
    min_child_weight=1e-3,  # min_sum_hessian_in_leaf   1e-3        [0, inf)    larger
    subsample=0.6,          # bagging_fraction          1.0         (0, 1]      smaller
    colsample_bytree=0.2,   # feature_fraction          1.0         (0, 1]      smaller
    colsample_bynode=0.2,   # feature_fraction_bynode   1.0         (0, 1]      smaller
    reg_alpha=0.03,         # lambda_l1                 0.0         [0, inf)    larger
    reg_lambda=0.3,         # lambda_l2                 0.0         [0, inf)    larger
    scale_pos_weight=1,     # ----                      1.0
    random_state=0,         # seed                      None
    n_jobs=-1,              # ----                      -1
    verbosity=1             # ----                      1           [0,1,2]
)
gbm.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], early_stopping_rounds=5, verbose=2)
y_pred = gbm.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))

# Random Search
param_grid = {'max_depth': [10, 20], 'num_leaves': [20, 30]}
rs = RandomizedSearchCV(gbm, param_grid, n_iter=3, cv=3, n_jobs=-1)
rs.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], early_stopping_rounds=3)
y_pred = rs.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))


## CatBoost Classifier
# https://catboost.ai/docs/concepts/python-reference_parameters-list.html
# https://catboost.ai/docs/concepts/parameter-tuning.html
X_trn, X_val, y_trn, y_val = train_test_split(X_tran, y_tran, train_size=0.9, random_state=2020)

# Train One Model
cat = CatBoostClassifier(   # alias                     default     range       prevents overfitting
    bootstrap_type='MVS',
    grow_policy='Lossguide',
    n_estimators=100,       # iterations                1000        [1, inf)    larger
    max_depth=20,           # depth                     6           [1, inf)    smaller
    learning_rate=0.3,      # eta                       0.1         (0, inf)    smaller
    max_bin=255,            # ----                      255         [2, inf)    smaller
    num_leaves=30,          # max_leaves                31          [2, 131072] smaller
    min_child_samples=2,    # min_data_in_leaf          1           [0, inf)    larger
    subsample=0.8,          # bagging_fraction          0.8         (0, 1]      smaller
    colsample_bylevel=0.2,  # rsm                       1.0         (0, 1]      smaller
    reg_lambda=0.3,         # l2_leaf_reg               3.0         [0, inf)    larger
    use_best_model=True,
    random_state=0,         # seed                      None
    thread_count=-1,        # ----                      -1
    verbose=1               # ----                      1           [0,1,2]
)
cat.fit(X_trn, y_trn, eval_set=(X_val, y_val), early_stopping_rounds=5, verbose=2)
y_pred = cat.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))


# Random Search
param_grid = {'max_depth': [10, 20], 'num_leaves': [20, 30]}
rs = RandomizedSearchCV(cat, param_grid, n_iter=3, cv=3, n_jobs=-1)
rs.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], early_stopping_rounds=3)
y_pred = rs.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))






