# import libraries
from sqlalchemy import create_engine
from sqlalchemy.types import String
import pandas as pd
import numpy as np
import sys
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)

import nltk
nltk.download(['punkt', 'wordnet'])
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def load_data(messages_filepath, categories_filepath):
    ''' Loads the 2 datasets and merges them in one table'''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df, categories

def clean_data(df, categories):
    '''Cleans and transforms dataset so that each
       category is represented by one column with binary values
       Drops duplicate entries as well'''

    categories = categories.categories.str.split(';', expand=True)
    category_colnames = categories.iloc[1].apply(lambda x: x[:-2]).to_list()
    categories = categories.applymap(lambda x: int(x[-1]))
    categories.columns = category_colnames
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], join="inner", axis=1)
    df.drop_duplicates(inplace=True)
    return  df

def save_data(df, database_filename):
    ''' Saves table in a SQL database'''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_msg', engine, if_exists='replace', index=False,  dtype={'message': String})
    return df


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
