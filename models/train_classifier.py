import sys
import sqlite3
import pandas as pd
import numpy as np
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)

import nltk
nltk.download(['punkt', 'wordnet'])
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle

def load_data(database_filepath):
    ''' Loads dataframe from SQL database and
    defines independant and dependat variables.
    Only the dependant variables that show more than
    one value are considered'''

    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM disaster_msg', con=conn)
    category_names = df.columns[df.nunique() > 1][4:]
    X = df['message']
    y = df[category_names]
    return X, y, category_names

def tokenize(text):
    ''' Prepares text data for input in the
    classifier. '''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    ''' Build the model to be used later for training
    and prediciting. Creates TFIDF version of text data
    and performs grid search  on model parameters'''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=7600))),
      ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, y_test):
    ''' Performs evaluation of the model.
    Prints f1_score, precision and recall score
    for all labels concerned'''

    y_pred = model.predict(X_test)
    y_pr = y_pred.transpose().tolist()
    y_true = y_test.values.transpose().tolist()
    f1_l = []
    precision = []
    recall = []
    for i, j in zip(y_pr, y_true):
        f1 = f1_score(i, j, average='macro', zero_division=0)
        f1_l.append(f1)
        prec = precision_score(i, j, average='macro')
        precision.append(prec)
        rec = recall_score(i, j, average='macro')
        recall.append(rec)
    print('f1 score for the 35 labels is {}'.format(f1_l))
    print('precision score for the 35 labels is {}'.format(precision))
    print('recall score for the 35 labels is {}'.format(recall))
    print("Best Parameters: {}".format(model.best_params_))


def save_model(model, model_filepath):
    ''' Saves model as pickle file
    to later call in the flask app'''

    with open(model_filepath, 'wb') as f:
         pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

