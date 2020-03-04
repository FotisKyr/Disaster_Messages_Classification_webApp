import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)

import nltk
nltk.download(['punkt', 'wordnet'])
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle

def load_data(database_filepath):
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM disaster_msg', con=conn)
    category_names = df.columns[df.nunique() > 1][4:]
    X = df['message']
    y = df[category_names]
    return df, X, y, category_names

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

df, X, y, category_names = load_data(r'C:\Users\Fotis\Desktop\Udacity\Data Scientist\Project_Disaster_Response_Pipelines\data\DisasterResponse.db')

'''df_value_counts = df.apply(pd.Series.value_counts)
df_value_counts.reset_index(inplace=True)
df_value_counts = df_value_counts.query('index==1')
df_value_counts = df_value_counts[category_names]
df_value_counts.sort_values(by=1, ascending=False, axis=1, inplace=True)

print(df_value_counts.columns[0:10])
row = df_value_counts.iloc[0,0:10]
plt.bar(x=df_value_counts.columns[0:10], height=list(row))
plt.show()
print(list(row))'''

df_a = df[category_names]
total = df_a.sum(numeric_only=True).sort_values(ascending=False)
print(total[-10:])
print(total.index[-10:])

plt.bar(x=total.index[-10:], height=total[-10:])
plt.show()




