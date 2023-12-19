import gc
import os
import itertools
import pickle
import re
import time
from random import choice, choices
from functools import reduce
from tqdm import tqdm
from itertools import cycle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from functools import reduce
from itertools import cycle
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn import metrics, model_selection, preprocessing, linear_model, ensemble, decomposition, tree
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
import re
import copy
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

'''
Reconstruction
'''

class EssayConstructor:
    
    def processingInputs(self,currTextInput):
        # Where the essay content will be stored
        essayText = ""
        # Produces the essay
        for Input in currTextInput.values:
            # Input[0] = activity
            # Input[1] = cursor_position
            # Input[2] = text_change
            # Input[3] = id
            # If activity = Replace
            if Input[0] == 'Replace':
                # splits text_change at ' => '
                replaceTxt = Input[2].split(' => ')
                # DONT TOUCH
                essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
                continue

            # If activity = Paste    
            if Input[0] == 'Paste':
                # DONT TOUCH
                essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
                continue

            # If activity = Remove/Cut
            if Input[0] == 'Remove/Cut':
                # DONT TOUCH
                essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
                continue

            # If activity = Move...
            if "M" in Input[0]:
                # Gets rid of the "Move from to" text
                croppedTxt = Input[0][10:]              
                # Splits cropped text by ' To '
                splitTxt = croppedTxt.split(' To ')              
                # Splits split text again by ', ' for each item
                valueArr = [item.split(', ') for item in splitTxt]              
                # Move from [2, 4] To [5, 7] = (2, 4, 5, 7)
                moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
                # Skip if someone manages to activiate this by moving to same place
                if moveData[0] != moveData[2]:
                    # Check if they move text forward in essay (they are different)
                    if moveData[0] < moveData[2]:
                        # DONT TOUCH
                        essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                    else:
                        # DONT TOUCH
                        essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
                continue                
                
            # If activity = input
            # DONT TOUCH
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
        return essayText
            
            
    def getEssays(self,df):
        # Copy required columns
        textInputDf = copy.deepcopy(df[['id', 'activity', 'cursor_position', 'text_change']])
        # Get rid of text inputs that make no change
        textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']     
        # construct essay, fast 
        tqdm.pandas()
        essay=textInputDf.groupby('id')[['activity','cursor_position', 'text_change']].progress_apply(lambda x: self.processingInputs(x))      
        # to dataframe
        essayFrame=essay.to_frame().reset_index()
        essayFrame.columns=['id','essay']
        # Returns the essay series
        return essayFrame
    

'''
Feat eng
'''

def split_and_compute_lengths(df, column, delimiter, new_column):
    """Split essays in a DataFrame by a delimiter, remove empty items, and compute lengths, word counts, and word count lists of paragraphs/sentences per id."""
    # Split the essays using the specified delimiter
    split_essays = df[column].str.split(delimiter)
    
    # Remove empty items (empty paragraphs/sentences)
    split_essays = split_essays.apply(lambda x: [item.strip() for item in x if item.strip()])
    
    # Compute the lengths of each paragraph/sentence
    lengths = split_essays.apply(lambda x: [len(paragraph) for paragraph in x])
    
    # Compute the word counts of each paragraph/sentence
    word_counts = split_essays.apply(lambda x: [len(paragraph.split()) for paragraph in x])
    
    # Create a new DataFrame to store the results
    result_df = pd.DataFrame({'id': df['id'], new_column+'_len': lengths, new_column+'_word_count': word_counts})
    
    return result_df

def compute_aggregations(df, id_col, agg_columns):
    """Computes specified aggregations for each id in the DataFrame using split_and_compute_lengths."""
    
    # Specify the aggregation functions
    agg_funcs = ['count', 'mean', 'median', 'std', 'min', 'max', 'var', 'sem']
    
    # Group by 'id' and compute the specified aggregations for the specified columns
    df = df.explode(agg_columns)
    agg_df = df.groupby(id_col).agg(agg_funcs)
    agg_df.columns = ['_'.join(col) for col in agg_df.columns]
    
    return agg_df.reset_index()

def calculate_relative_paragraph_sizes(input_df, essay_column):

    df = input_df.copy()
    # Split the essay text into paragraphs
    df['paragraphs'] = df[essay_column].str.split('\n')

    # Filter out empty paragraphs
    df['paragraphs'] = df['paragraphs'].apply(lambda paragraphs: [p for p in paragraphs if p.strip() != ''])

    # Calculate the total number of paragraphs
    df['total_paragraphs'] = df['paragraphs'].apply(len)

    # Calculate the relative sizes
    df['relative_intro_size'] = 1 / df['total_paragraphs']  # First paragraph is the introduction
    df['relative_body_size'] = (df['total_paragraphs'] - 2) / df['total_paragraphs']  # Middle paragraphs are the body
    df['relative_conclusion_size'] = 1 / df['total_paragraphs']  # Last paragraph is the conclusion

    # Calculate the word count for each paragraph
    df['paragraph_word_count'] = df['paragraphs'].apply(lambda x: [len(paragraph.split()) for paragraph in x])

    # Separate paragraphs into intro, body, and conclusion
    df['word_count_intro'] = df['paragraph_word_count'].apply(lambda x: x[0] if len(x) > 0 else 0)
    df['word_count_body'] = df['paragraph_word_count'].apply(lambda x: sum(x[1:-1]) if len(x) > 2 else 0)
    df['word_count_conclusion'] = df['paragraph_word_count'].apply(lambda x: x[-1] if len(x) > 1 else 0)

    # Calculate total word count for each essay
    df['total_word_count'] = df['paragraph_word_count'].apply(sum)
    
    # Calculate ratios
    df['intro_ratio'] = df['word_count_intro'] / df['total_word_count']
    df['body_ratio'] = df['word_count_body'] / df['total_word_count']
    df['conclusion_ratio'] = df['word_count_conclusion'] / df['total_word_count']
    
    df['intro_body_ratio'] = df['word_count_intro'] / df['word_count_body']
    df['intro_conclusion_ratio'] = df['word_count_intro'] / df['word_count_conclusion']
    df['body_conclusion_ratio'] = df['word_count_body'] / df['word_count_conclusion']


    # Drop intermediate columns if needed
    df.drop(columns=['paragraphs', 'total_paragraphs', essay_column, 'paragraph_word_count'], inplace=True)

    return df

class FeatureEngineering:

    @staticmethod
    def get_input_words(df: pd.DataFrame) -> pd.DataFrame:
        """Extracts and aggregates information about input words from the text changes in the dataset."""
        # Filter relevant rows and reset index
        filtered_df = df[(~df['text_change'].str.contains('=>')) & (df['text_change'] != 'NoChange')].reset_index(drop=True)
    
        # Group and concatenate text changes
        grouped_df = filtered_df.groupby('id')['text_change'].apply(''.join).reset_index()
    
        # Find all occurrences of 'q+'
        grouped_df['input_words'] = grouped_df['text_change'].apply(lambda x: re.findall(r'q+', x))
    
        # Calculate various statistics
        stats_df = grouped_df['input_words'].apply(lambda words: pd.Series({
            'input_word_count': len(words),
            'input_word_length_mean': np.mean([len(word) for word in words]) if words else 0,
            'input_word_length_max': np.max([len(word) for word in words]) if words else 0,
            'input_word_length_std': np.std([len(word) for word in words]) if words else 0
        }))

        return pd.concat([grouped_df[['id']], stats_df], axis=1)

class FeatureStats:

    @staticmethod
    def get_word_counts(df: pd.DataFrame, id_col: str, word_count_col: str) -> pd.DataFrame:
        """Aggregates the final word count for each essay."""
        return df.groupby(id_col).agg(final_word_count=(word_count_col, 'last'))


class Preprocessor:

    def __init__(self):
        self.activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
        self.text_changes = ['q', ' ', 'NoChange', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
        self.count_vect = CountVectorizer()
        self.tfidf_vect = TfidfVectorizer()
        self.idf = {}

    def activity_counts(self, df):
        """Calculates activity counts for each essay."""
        activity_counts = df.groupby('id')['activity'].value_counts().unstack(fill_value=0)
        activity_counts = activity_counts.reindex(columns=self.activities, fill_value=0)

        # Apply IDF scaling if needed
        if not self.idf:
            self.idf = {col: np.log(df.shape[0] / (activity_counts[col].sum() + 1)) for col in self.activities}
        activity_counts = activity_counts.apply(lambda x: (1 + np.log(x)) * self.idf.get(x.name, 0), axis=0)

        return activity_counts.add_prefix('activity_')

    def event_counts(self, df, colname):
        """Calculates event counts for each essay."""
        events = ['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp', 'CapsLock', 
                  "'", 'Delete', 'Unidentified']

        event_counts = df.groupby('id')[colname].value_counts().unstack(fill_value=0)
        event_counts = event_counts.reindex(columns=events, fill_value=0)

        # Apply IDF scaling if needed
        if not self.idf:
            self.idf = {col: np.log(df.shape[0] / (event_counts[col].sum() + 1)) for col in events}
        event_counts = event_counts.apply(lambda x: (1 + np.log(x)) * self.idf.get(x.name, 0), axis=0)

        return event_counts.add_prefix(f'{colname}_')


    def text_change_counts(self, df):
        """Calculates counts of different types of text changes for each essay."""
        text_change_counts = df.groupby('id')['text_change'].value_counts().unstack(fill_value=0)
        text_change_counts = text_change_counts.reindex(columns=self.text_changes, fill_value=0)

        return text_change_counts.add_prefix('text_change_')
    
    def match_punctuations(self, df):
        """Counts the number of punctuation marks used in each essay."""
        punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/',
                        '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']

        # Filter the DataFrame to include only rows with punctuation events
        punctuation_df = df[df['down_event'].isin(punctuations)]

        # Group by 'id' and 'down_event' and count the occurrences of each punctuation
        punctuation_counts = punctuation_df.groupby(['id', 'down_event'])['down_event'].count().unstack(fill_value=0)

        # Calculate the total punctuation count for each 'id'
        total_punctuation_counts = punctuation_counts.sum(axis=1)

        # Add the total count as a new column
        punctuation_counts['Total'] = total_punctuation_counts

        return punctuation_counts

    def compute_time_gaps(self, df, gap_list):
        """Computes time gaps between events for a list of specified gaps."""
        for gap in gap_list:
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']

        time_gap_cols = [f'action_time_gap{gap}' for gap in gap_list]
        return df[['id'] + time_gap_cols].groupby('id').agg(['mean', 'std', 'min', 'max'])

    @staticmethod
    def count_pauses(group):
        """Counts pauses longer than 2000 ms."""
        gap = group['down_time'] - group['up_time'].shift(1)
        return (gap > 2000).sum()

    @staticmethod
    def pause_proportion(group):
        """Calculates the proportion of pause time to total essay time."""
        gap = group['down_time'] - group['up_time'].shift(1)
        total_pause_time = gap[gap > 2000].sum()
        total_essay_time = group['up_time'].max() - group['down_time'].min()
        return total_pause_time / total_essay_time if total_essay_time else 0

    @staticmethod
    def mean_pause_length(group):
        """Calculates the mean length of pauses longer than 2000 ms."""
        gap = group['down_time'] - group['up_time'].shift(1)
        pauses = gap[gap > 2000]
        return pauses.mean() / 1000 if not pauses.empty else 0

    # Method to aggregate all pause-related features
    def aggregate_pause_features(self, df):
        """Aggregates all pause-related features for each essay."""
        grouped = df.groupby('id')
        pause_features = pd.DataFrame()
        pause_features['n_pauses'] = grouped.apply(self.count_pauses)
        pause_features['pause_proportion'] = grouped.apply(self.pause_proportion)
        pause_features['mean_pause_length'] = grouped.apply(self.mean_pause_length)
        return pause_features

    @staticmethod
    def process_variance(group):
        """Calculates the variance in the writing process over time for each essay."""
        if len(group) < 2:  # Handling for groups with a single row
            return 0

        bins = np.linspace(group['down_time'].min(), group['up_time'].max(), 11)
        divisions = pd.cut(group['down_time'], bins=bins, include_lowest=True, labels=range(1, 11))
        production_deciles = group.groupby(divisions).agg(n_events=('event_id', 'count'))
        return np.std(production_deciles['n_events'], ddof=1)

    def aggregate_process_variance(self, df):
        """Aggregates the process variance feature for each essay."""
        return df.groupby('id').apply(self.process_variance).rename('process_variance').to_frame()

    def create_time_features(self, df):
        """Generates aggregated time-related features for each essay ID."""
        df = df.copy()
        # Time-based calculations
        df['action_time_sec'] = (df['up_time'] - df['down_time']) / 1000.0
        df['time_since_last_event'] = df.groupby('id')['down_time'].diff() / 1000.0
        df['cumulative_action_time'] = df.groupby('id')['action_time_sec'].cumsum()

        # Prepare aggregation dictionary
        aggregations = {
            'action_time_sec': ['mean', 'sum', 'max', 'std'],
            'time_since_last_event': ['mean', 'max', 'std'],
            'cumulative_action_time': ['max']
        }

        # Add rolling window features to aggregations
        for window in [5, 10, 15, 20, 30, 50]:
            df[f'rolling_mean_{window}'] = df.groupby('id')['action_time_sec'].transform(lambda x: x.rolling(window).mean())
            df[f'rolling_std_{window}'] = df.groupby('id')['action_time_sec'].transform(lambda x: x.rolling(window).std())
            aggregations[f'rolling_mean_{window}'] = ['mean']
            aggregations[f'rolling_std_{window}'] = ['mean']

        # Aggregating features for each ID
        aggregated_features = df.groupby('id').agg(aggregations)
        aggregated_features.columns = ['_'.join(col) for col in aggregated_features.columns]
        return aggregated_features.reset_index()

    def create_additional_time_features(self, df):
        """Generates additional aggregated time features for each essay ID."""
        df = df.copy()
        df['time_diff'] = abs(df.groupby('id')['down_time'].diff() - df['up_time'].shift(1)) / 1000
        df['time_diff'] = df['time_diff'].fillna(0)  # Handling the first row for each ID

        # Prepare aggregation dictionary
        aggregates = {
            'time_diff': ['max', 'median']  # Initial pause as the first value
        }

        # Adding boolean counts for pauses
        for pause in [0.5, 1, 1.5, 2, 3, 5, 10, 20]:
            df[f'pauses_{pause}_sec'] = df['time_diff'].apply(lambda x: x > pause)
            aggregates[f'pauses_{pause}_sec'] = ['sum']

        # Aggregating features for each ID
        additional_features = df.groupby('id').agg(aggregates)
        additional_features.columns = ['_'.join(col) for col in additional_features.columns]
        return additional_features.reset_index()


    def make_text_features(self, df, column='text_change', fit_transform=True):
        """Extracts text features using CountVectorizer and TfidfVectorizer, along with custom features."""
        # Filter and concatenate text changes
        filtered_df = df[(~df[column].str.contains('=>')) & (df[column] != 'NoChange')]
        concatenated_texts = filtered_df.groupby('id')[column].apply(' '.join).reset_index()

        # Apply CountVectorizer and TfidfVectorizer
        if fit_transform:
            bow_features = self.count_vect.fit_transform(concatenated_texts[column])
            tfidf_features = self.tfidf_vect.fit_transform(concatenated_texts[column])
        else:
            bow_features = self.count_vect.transform(concatenated_texts[column])
            tfidf_features = self.tfidf_vect.transform(concatenated_texts[column])

        # Convert to DataFrame
        bow_df = pd.DataFrame(bow_features.toarray(), columns=[f'bow_{name}' for name in self.count_vect.get_feature_names_out()])
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=[f'tfidf_{name}' for name in self.tfidf_vect.get_feature_names_out()])

        # Custom Feature: Length of each essay
        custom_features_df = pd.DataFrame({'custom_length': concatenated_texts[column].apply(len)})

        # Merge all features
        merged_features = pd.concat([concatenated_texts[['id']], bow_df, tfidf_df, custom_features_df], axis=1)
        return merged_features

    def compute_cursor_position_change_features(self, df, gap_list):
        """Computes cursor position change features for specified gaps per 'id'."""
        result = pd.DataFrame()  # Create an empty DataFrame to store the results
        for gap in gap_list:
            col_shift = f'cursor_position_shift{gap}'
            df[col_shift] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[col_shift]
            df[f'cursor_position_abs_change{gap}'] = abs(df[f'cursor_position_change{gap}'])
            # Aggregate the results per 'id'
            id_features = df.groupby('id').agg({
                f'cursor_position_change{gap}': 'mean',  # You can choose different aggregation functions as needed
                f'cursor_position_abs_change{gap}': 'mean'
            }).reset_index()
            result = pd.concat([result, id_features], axis=1)  # Concatenate the results

        result = result.loc[:, ~result.columns.duplicated()]  # Remove duplicate columns
        return result

    def compute_word_count_change_features(self, df, gap_list):
        """Computes word count change features for specified gaps per 'id'."""
        result = pd.DataFrame()  # Create an empty DataFrame to store the results
        for gap in gap_list:
            col_shift = f'word_count_shift{gap}'
            df[col_shift] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[col_shift]
            df[f'word_count_abs_change{gap}'] = abs(df[f'word_count_change{gap}'])
            # Aggregate the results per 'id'
            id_features = df.groupby('id').agg({
                f'word_count_change{gap}': 'mean',  # You can choose different aggregation functions as needed
                f'word_count_abs_change{gap}': 'mean'
            }).reset_index()
            result = pd.concat([result, id_features], axis=1)  # Concatenate the results

        result = result.loc[:, ~result.columns.duplicated()]  # Remove duplicate columns
        return result

    # def compute_ratio_based_features(self, df):
    #     """Computes various ratio-based features per 'id'."""
    #     result = df.groupby('id').agg({
    #         'word_time_ratio': 'mean',  # You can choose different aggregation functions as needed
    #         'word_event_ratio': 'mean',
    #         'event_time_ratio': 'mean'
    #     }).reset_index()
    #     return result

    # ('word_count', ['nunique', 'max', 'quantile', 'sem', 'mean'])] It needs word_count_max to work
    
    def get_keyboard_mouse_feats(self, train_logs_df):

        # Creates two shift variables which lag the original variable by 1 and 2 periods respectively.
        event_df = train_logs_df[['id', 'event_id', 'down_event']].copy(deep=True)

        event_df['down_event_shift_1'] = event_df['down_event'].shift(periods=1)
        event_df['down_event_shift_2'] = event_df['down_event'].shift(periods=2)

        event_df = event_df[['id', 'event_id', 'down_event_shift_2', 'down_event_shift_1', 'down_event']]

        ctrl_bksp_df = ((event_df['down_event_shift_1'] == 'Control') & (event_df['down_event'] == 'Backspace')).groupby(event_df['id']).sum().reset_index(name='count')
        ctrl_c_df = ((event_df['down_event_shift_1'] == 'Control') & (event_df['down_event'].str.lower() == 'c')).groupby(event_df['id']).sum().reset_index(name='count')
        ctrl_v_df = ((event_df['down_event_shift_1'] == 'Control') & (event_df['down_event'].str.lower() == 'v')).groupby(event_df['id']).sum().reset_index(name='count')
        ctrl_x_df = ((event_df['down_event_shift_1'] == 'Control') & (event_df['down_event'].str.lower() == 'x')).groupby(event_df['id']).sum().reset_index(name='count')

        # Creating a DataFrame that contains all counts at an id level.

        kb_shortcut_df = pd.DataFrame(event_df['id'].unique(), columns=['id'])

        kb_shortcut_df['ctrl_bksp_cnt'] = ctrl_bksp_df['count']
        kb_shortcut_df['ctrl_c_cnt'] = ctrl_c_df['count']
        kb_shortcut_df['ctrl_v_cnt'] = ctrl_v_df['count']
        kb_shortcut_df['ctrl_x_cnt'] = ctrl_x_df['count']

        mouse_event_df = pd.DataFrame(train_logs_df['id'].unique(), columns=['id'])

        # # Calculating the proportion of mouse click events
        mouse_event_df['mouse_event_cnt'] = train_logs_df.groupby(train_logs_df['id'])['down_event'].apply(lambda x: (x.isin(['Leftclick', 'Rightclick', 'Middleclick', 'Unknownclick']).sum())).reset_index()['down_event']

        mouse_event_df['all_event_cnt'] = train_logs_df.groupby(train_logs_df['id'])['event_id'].max().reset_index()['event_id']

        mouse_event_df['mouse_event_perc'] = (mouse_event_df['mouse_event_cnt']/mouse_event_df['all_event_cnt'])*100.0
        
        return kb_shortcut_df.merge(mouse_event_df, on='id')