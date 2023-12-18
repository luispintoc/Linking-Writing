import tqdm
import pandas as pd
import copy
from collections import defaultdict
from tqdm import tqdm
tqdm.pandas()
import re
from collections import Counter
import numpy as np
from scipy.stats import kurtosis, skew, hmean
from scipy.stats.mstats import winsorize
from numpy import percentile
from scipy.stats import gmean, trim_mean, entropy
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


# Helper functions

def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)

AGGREGATIONS = [
    'count',  # Count of non-null values
    'mean',  # Mean (average)
    'median',  # Median (middle value)
    'std',  # Standard deviation
    'min',  # Minimum value
    'max',  # Maximum value
    'sum',  # Sum of values
    'var',  # Variance
    'sem', # Standard error of the mean
    'first',  # First value
    'last',  # Last value
]


def split_essays_into_sentences(df):
    essay_df = df
    essay_df['id'] = essay_df.index
    essay_df['sent'] = essay_df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    essay_df = essay_df.explode('sent')
    essay_df['sent'] = essay_df['sent'].apply(lambda x: x.replace('\n','').strip())
    # Number of characters in sentences
    essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    essay_df['sent_word_count'] = essay_df['sent'].apply(lambda x: len(x.split(' ')))
    essay_df = essay_df[essay_df.sent_len!=0].reset_index(drop=True)
    return essay_df

def compute_sentence_aggregations(df):
    sent_agg_df = pd.concat(
        [df[['id','sent_len']].groupby(['id']).agg(AGGREGATIONS), df[['id','sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1
    )
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
    return sent_agg_df

def split_essays_into_paragraphs(df):
    essay_df = df
    essay_df['id'] = essay_df.index
    essay_df['paragraph'] = essay_df['essay'].apply(lambda x: x.split('\n'))
    essay_df = essay_df.explode('paragraph')
    # Number of characters in paragraphs
    essay_df['paragraph_len'] = essay_df['paragraph'].apply(lambda x: len(x)) 
    # Number of words in paragraphs
    essay_df['paragraph_word_count'] = essay_df['paragraph'].apply(lambda x: len(x.split(' ')))
    essay_df = essay_df[essay_df.paragraph_len!=0].reset_index(drop=True)
    return essay_df

def compute_paragraph_aggregations(df):
    paragraph_agg_df = pd.concat(
        [df[['id','paragraph_len']].groupby(['id']).agg(AGGREGATIONS), df[['id','paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1
    ) 
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return paragraph_agg_df

class Preprocessor:
    
    def __init__(self, seed):
        self.seed = seed
        
        self.activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
        self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 
              'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
        self.text_changes = ['q', ' ', 'NoChange', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/',
                        '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']
        self.gaps = [1, 2, 3, 5, 10, 20, 50, 100]
        
        self.idf = defaultdict(float)
    
    def activity_counts(self, df):
        tmp_df = df.groupby('id').agg({'activity': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['activity'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.activities:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'activity_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def event_counts(self, df, colname):
        tmp_df = df.groupby('id').agg({colname: list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df[colname].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.events:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'{colname}_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf
            
            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def text_change_counts(self, df):
        tmp_df = df.groupby('id').agg({'text_change': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['text_change'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.text_changes:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'text_change_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf
            
            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf
            
        return ret

    def match_punctuations(self, df):
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['down_event'].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret})
        return ret

    def get_input_words(self, df):
        tmp_df = df[(~df['text_change'].str.contains('=>'))&(df['text_change'] != 'NoChange')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: ''.join(x))
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: re.findall(r'q+', x))
        tmp_df['input_word_count'] = tmp_df['text_change'].apply(len)
        tmp_df['input_word_length_mean'] = tmp_df['text_change'].apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_max'] = tmp_df['text_change'].apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_std'] = tmp_df['text_change'].apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df.drop(['text_change'], axis=1, inplace=True)
        return tmp_df
    
    def make_feats(self, df):
        
        feats = pd.DataFrame({'id': df['id'].unique().tolist()})
        
        print("Engineering time data")
        for gap in self.gaps:
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']
        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering cursor position data")
        for gap in self.gaps:
            df[f'cursor_position_shift{gap}'] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[f'cursor_position_shift{gap}']
            df[f'cursor_position_abs_change{gap}'] = np.abs(df[f'cursor_position_change{gap}'])
        df.drop(columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering word count data")
        for gap in self.gaps:
            df[f'word_count_shift{gap}'] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[f'word_count_shift{gap}']
            df[f'word_count_abs_change{gap}'] = np.abs(df[f'word_count_change{gap}'])
        df.drop(columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)
        
        print("Engineering statistical summaries for features")
        feats_stat = [
            ('event_id', ['max']),
            ('up_time', ['max']),
            ('action_time', AGGREGATIONS),
            ('activity', ['nunique']),
            ('down_event', ['nunique']),
            ('up_event', ['nunique']),
            ('text_change', ['nunique']),
            ('cursor_position', ['nunique', 'max', 'quantile', 'sem', 'mean']),
            ('word_count', ['nunique', 'max', 'quantile', 'sem', 'mean'])]
        for gap in self.gaps:
            feats_stat.extend([
                (f'action_time_gap{gap}', AGGREGATIONS),
                (f'cursor_position_change{gap}', AGGREGATIONS),
                (f'word_count_change{gap}', AGGREGATIONS)
            ])
        
        pbar = tqdm(feats_stat)
        for item in pbar:
            colname, methods = item[0], item[1]
            for method in methods:
                pbar.set_postfix()
                if isinstance(method, str):
                    method_name = method
                else:
                    method_name = method.__name__
                pbar.set_postfix(column=colname, method=method_name)
                tmp_df = df.groupby(['id']).agg({colname: method}).reset_index().rename(columns={colname: f'{colname}_{method_name}'})
                feats = feats.merge(tmp_df, on='id', how='left')

        print("Engineering activity counts data")
        tmp_df = self.activity_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering event counts data")
        tmp_df = self.event_counts(df, 'down_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        tmp_df = self.event_counts(df, 'up_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering text change counts data")
        tmp_df = self.text_change_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering punctuation counts data")
        tmp_df = self.match_punctuations(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering input words data")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        print("Engineering ratios data")
        feats['word_time_ratio'] = feats['word_count_max'] / feats['up_time_max']
        feats['word_event_ratio'] = feats['word_count_max'] / feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max']  / feats['up_time_max']
        feats['idle_time_ratio'] = feats['action_time_gap1_sum'] / feats['up_time_max']

        return feats
    
'''
Feats from: https://www.kaggle.com/code/magnussesodia/writing-processes-feature-generation
'''

def get_word_counts(train):
    word_counts = train.groupby(['id']).agg(final_word_count=('word_count', 'last'))
    return word_counts

def get_session_length(sub_df):
    start = np.clip(sub_df['down_time'].min(), 0, 1.8e6)
    end = np.clip(sub_df['up_time'].max(), 0, 1.8e6) # clip times to within the 30 minute window, since there is some anomalous data
    
    session_length_in_minutes = (end - start) * (1 / 1000) * (1 / 60)
    return session_length_in_minutes

def get_keys_pressed_per_minute(train):
    inputs_remove_cut = train[train['activity'].isin(['Input', 'Remove/Cut'])]
    keys_pressed_per_minute = inputs_remove_cut.groupby(['id']).agg(keys_pressed_per_minute=('event_id', 'count'))
    session_lengths = inputs_remove_cut.groupby(['id']).apply(get_session_length)
    
    keys_pressed_per_minute['keys_pressed_per_minute'] = round(keys_pressed_per_minute['keys_pressed_per_minute'] / session_lengths, 2)
    
    return keys_pressed_per_minute

# Define a pause as a period of time between the up_time of one event and the down_time of the following event that is greater than 2000 ms

# helper
def count_pauses(sub_df):
    gap = sub_df['down_time'] - sub_df['up_time'].shift(1)
    pauses = (gap > 2000).sum()
    return pauses

def get_n_pauses(train):
    n_pauses = train.groupby(['id']).apply(count_pauses).rename('n_pauses')
    return n_pauses

# helper
def pause_proportion(sub_df):
    gap = sub_df['down_time'] - sub_df['up_time'].shift(1)
    total_pause_time = gap[gap > 2000].sum()
    total_essay_time = sub_df['up_time'].max() - sub_df['down_time'].min()
    proportion = round(total_pause_time / total_essay_time, 4)
    
    return proportion

def get_pause_proportions(train):
    pause_proportions = train.groupby(['id']).apply(pause_proportion).rename('pause_proportion')
    return pause_proportions

# helper
def mean_pause_length(sub_df):
    gap = sub_df['down_time'] - sub_df['up_time'].shift(1)
    mean_pause_length = round(gap[gap > 2000].mean() / 1000, 2)
    
    return mean_pause_length

def get_mean_pause_lengths(train):
    mean_pause_lengths = train.groupby(['id']).apply(mean_pause_length).rename('mean_pause_length')
    return mean_pause_lengths

# helper
def process_variance(sub_df):
    bins = np.linspace(sub_df['down_time'].min(), sub_df['up_time'].max(), 11)
    divisions = pd.cut(sub_df['down_time'], bins=bins, include_lowest=True, labels=range(1, 11))
    production_deciles = sub_df.groupby(divisions).agg(n_events=('event_id', 'count'))
    process_variance = np.std(production_deciles['n_events'], ddof=1)
    return process_variance

def get_process_variances(train):
    process_variances = train.groupby(['id']).apply(process_variance).rename('process_variance')
    return process_variances

def gen_features(train):
    ids = pd.DataFrame({'id': train['id'].unique()}).set_index('id')
    
    word_counts = get_word_counts(train)
    keys_pressed_per_minute = get_keys_pressed_per_minute(train)
    n_pauses = get_n_pauses(train)
    pause_proportions = get_pause_proportions(train)
    mean_pause_lengths = get_mean_pause_lengths(train)
    process_variances = get_process_variances(train)

    X = pd.concat([ids,
                   word_counts,
                   keys_pressed_per_minute,
                   n_pauses,
                   pause_proportions,
                   mean_pause_lengths,
                   process_variances], axis=1)
    
    X = X.reset_index()
    
    return X



'''
https://www.kaggle.com/code/habedi/baseline-model-with-over-2000-features#2.2-Generating-new-features
'''

# Keeping the states of these objects global to reuse them in the test data
count_vect = CountVectorizer()
tfidf_vect = TfidfVectorizer()

def make_text_features(df, name="Train Logs"):
        
    # Filter and reset index
    filtered_df = df[(~df['text_change'].str.contains('=>')) & (df['text_change'] != 'NoChange')].reset_index(drop=True)

    # Group and concatenate text changes
    grouped_df = filtered_df.groupby('id')['text_change'].apply(''.join).reset_index()

    # Find all occurrences of 'q+'
    grouped_df['text_change'] = grouped_df['text_change'].apply(lambda x: re.findall(r'q+', x))

    #tmp_df = df.groupby('id').agg({'text_change': list}).reset_index()
    
    tmp_df = grouped_df
    
    tmp_df['essay'] = tmp_df['text_change'].apply(lambda x: ' '.join(x))
    
    tmp_df.drop(columns=['text_change'], inplace=True)

    def count_encoding(essays_as_string, name=name):
        """Applies Count Encoding to the essay data and returns a DataFrame with prefixed column names."""
        
        if name == "Train Logs":
            features = count_vect.fit_transform(essays_as_string)
        else:
            features = count_vect.transform(essays_as_string)
            
        feature_names = [f'bow_{name}' for name in count_vect.get_feature_names_out()]
        return pd.DataFrame(features.toarray(), columns=feature_names)

    def tfidf_encoding(essays_as_string, name=name):
        """Applies TF-IDF Encoding to the essay data and returns a DataFrame with prefixed column names."""
        
        if name == "Train Logs":
            features = tfidf_vect.fit_transform(essays_as_string)
        else:
            features = tfidf_vect.transform(essays_as_string)
        
        feature_names = [f'tfidf_{name}' for name in tfidf_vect.get_feature_names_out()]
        return pd.DataFrame(features.toarray(), columns=feature_names)
    
    def custom_feature_engineering(essays):
        """Example custom feature: calculates the length of each essay with prefixed column name."""
        return pd.DataFrame({'custom_length': [len(essay) for essay in essays]})
    
    def merge_features(data):
        """Merges features from different methods into one DataFrame with the id column."""
        essays_as_string = data['essay']

        # Extract features
        bow_df = count_encoding(essays_as_string)
        tfidf_df = tfidf_encoding(essays_as_string)
        custom_features_df = custom_feature_engineering(data['essay'])
    
        # Merge all features
        merged_features = pd.concat([data[['id']], bow_df, tfidf_df, custom_features_df], axis=1)
        return merged_features
    
    return merge_features(tmp_df)

def create_time_features(dfx):
    
    df = dfx.copy()
    new_columns = []

    # Calculate action time in seconds
    df['action_time_sec'] = (df['up_time'] - df['down_time']) / 1000.0
    new_columns.append('action_time_sec')

    # Time since last event
    df['time_since_last_event'] = df.groupby('id')['down_time'].diff() / 1000.0
    new_columns.append('time_since_last_event')

    # Cumulative time of actions for each essay
    df['cumulative_action_time'] = df.groupby('id')['action_time_sec'].cumsum()
    new_columns.append('cumulative_action_time')

    # Time differences between down and up events
    for lag in [1, 2, 3, 5, 10]:
        col_down = f'time_diff_down_{lag}'
        col_up = f'time_diff_up_{lag}'
        df[col_down] = df.groupby('id')['down_time'].diff(periods=lag) / 1000.0
        df[col_up] = df.groupby('id')['up_time'].diff(periods=lag) / 1000.0
        new_columns.extend([col_down, col_up])

    # Count of events per essay
    df['event_count'] = df.groupby('id')['event_id'].transform('count')
    new_columns.append('event_count')

    # Average, max, min, and std of action time per essay
    stats_features = ['mean_action_time', 'max_action_time', 'min_action_time', 'std_action_time']
    for feature in stats_features:
        df[feature] = df.groupby('id')['action_time_sec'].transform(feature.split('_')[0])
        new_columns.append(feature)

    # Total word count per essay
    df['total_word_count'] = df.groupby('id')['word_count'].transform('max')
    new_columns.append('total_word_count')

    # Rolling window features (e.g., rolling mean and std of action times)
    window_sizes = [5, 10, 15, 20, 30, 50]
    for window in window_sizes:
        rolling_mean = f'rolling_mean_{window}'
        rolling_std = f'rolling_std_{window}'
        df[rolling_mean] = df.groupby('id')['action_time_sec'].transform(lambda x: x.rolling(window).mean())
        df[rolling_std] = df.groupby('id')['action_time_sec'].transform(lambda x: x.rolling(window).std())
        new_columns.extend([rolling_mean, rolling_std])

    return df, new_columns

def create_additional_time_features(dfx):
    # Copying the original DataFrame
    df = dfx.copy()

    # Creating new features
    df['up_time_lagged'] = df.groupby('id')['up_time'].shift(1).fillna(df['down_time'])
    df['time_diff'] = abs(df['down_time'] - df['up_time_lagged']) / 1000

    # Grouping by 'id'
    grouped = df.groupby('id')

    # Aggregating and computing new features
    aggregates = {
        'largest_latency': grouped['time_diff'].max(),
        'smallest_latency': grouped['time_diff'].min(),
        'median_latency': grouped['time_diff'].median(),
        'initial_pause': grouped['down_time'].first() / 1000
    }

    # Adding boolean counts for pauses
    for pause in [0.5, 1, 1.5, 2, 3, 5, 10, 20]:
        aggregates[f'pauses_{pause}_sec'] = grouped['time_diff'].apply(lambda x: ((x > pause) & (x < pause + 0.5)).sum())

    # Creating a new DataFrame from the aggregates
    features_df = pd.DataFrame(aggregates)

    # Getting the names of the new columns
    new_columns = features_df.columns.tolist()

    return features_df, new_columns

# Used some code from: https://www.kaggle.com/code/olyatsimboy/towards-tf-idf-in-logs-features

class FeatureMaker:

    def __init__(self):
        """Initializes the FeatureMaker class."""
        
        # Punctuation characters
        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/',
                        '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']

        # For making gap-based features        
        self.gaps = [1, 2, 3, 4, 5, 10, 20, 50, 100]

        # Original features to be used for feature engineering 
        self.categorical_features = ['activity', 'down_event', 'up_event', 'text_change']
        self.discrete_numeric_features = ['event_id', 'cursor_position', 'word_count', 'down_time', 'up_time',
                                          'action_time']

        # Aggregation functions to be used for feature engineering of continuous features
        self.aggregation_functions = [
            'count',  # Count of non-null values
            'mean',  # Mean (average)
            'median',  # Median (middle value)
            'std',  # Standard deviation
            'min',  # Minimum value
            'max',  # Maximum value
            'sum',  # Sum of values
            'var',  # Variance
            lambda x: x.quantile(0.25),  # 25th percentile
            lambda x: x.quantile(0.75),  # 75th percentile
            lambda x: x.max() - x.min(),  # Range (max - min)
            lambda x: x.nunique(),  # Number of unique values
            lambda x: x.mode()[0] if not x.mode().empty else np.NaN,  # Mode (most frequent value)
            'first',  # First value
            'last',  # Last value
            lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.NaN,  # Coefficient of variation
            lambda x: x.quantile(0.75) - x.quantile(0.25),  # Interquartile range
            lambda x: np.sum(x ** 2),  # Sum of squares
            lambda x: gmean(x.dropna()),  # Geometric mean
            lambda x: np.prod(x),  # Product of values
            lambda x: np.sqrt(np.mean(x ** 2)),  # Root mean square
            lambda x: trim_mean(x, 0.1),  # Trimmed mean (10% trimmed)
            lambda x: x.cummax().iloc[-1],  # Cumulative maximum
            lambda x: x.cummin().iloc[-1],  # Cumulative minimum
            lambda x: entropy(x.value_counts(normalize=True), base=2),  # Entropy
            # Additional aggregation functions
            'skew',  # Skewness
            kurtosis,  # Kurtosis
            hmean,  # Harmonic mean
            lambda x: percentile(x, 10),  # 10th percentile
            lambda x: percentile(x, 90),  # 90th percentile
            lambda x: np.mean(np.diff(x)),  # Mean of differences
            lambda x: np.median(np.abs(x - np.median(x))),  # Median absolute deviation
            lambda x: np.max(np.abs(x)),  # Max absolute value
            lambda x: np.min(np.abs(x)),  # Min absolute value
            lambda x: np.mean(np.abs(x)),  # Mean absolute value
            lambda x: np.var(np.abs(x)),  # Variance of absolute values
            lambda x: np.std(np.abs(x)),  # Standard deviation of absolute values
            lambda x: skew(x.dropna()),  # Skewness with NaN handling
            lambda x: kurtosis(x.dropna()),  # Kurtosis with NaN handling
            lambda x: winsorize(x, limits=[0.05, 0.05]).mean(),  # Winsorized mean
            lambda x: np.ptp(x),  # Peak to peak (max - min)
            lambda x: np.product(np.unique(x)),  # Product of unique values
            lambda x: np.sum(np.unique(x)),  # Sum of unique values
            lambda x: np.mean(x) / np.std(x) if np.std(x) != 0 else np.NaN,  # Signal to noise ratio
            lambda x: np.sqrt(np.var(x)),  # Root of variance
            lambda x: np.log(np.sum(np.exp(x)))  # Log-sum-exp
        ]

        # Aggregation functions to be used for feature engineering of categorical features
        self.categorical_aggregations = [
            'count',  # Count of non-null entries
            lambda x: x.value_counts().max(),  # Count of the most frequent value (mode)
            'nunique',  # Number of unique values
            lambda x: entropy(x.value_counts(normalize=True), base=2) if x.nunique() > 1 else 0,  # Entropy
            lambda x: x.value_counts(normalize=True).max(),  # Percentage of the most common value
            lambda x: x.value_counts(normalize=True).nlargest(2).iloc[-1] if x.nunique() > 1 else 0,
            # Percentage of the second most common value
            lambda x: len(x) - x.nunique(),  # Redundancy count (total count minus unique count)
            lambda x: (x == x.mode()[0]).sum() if not x.mode().empty else 0,  # Count of occurrences of the mode
            lambda x: (x == x.shift()).sum(),  # Count of consecutive duplicate values
            lambda x: x.apply(lambda v: len(str(v))).max(),  # Max length of the string representation of the categories
            lambda x: x.apply(lambda v: len(str(v))).min()  # Min length of the string representation of the categories
        ]

    def match_punctuations(self, df):
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['down_event'].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret})
        return ret

    def get_input_words(self, df):
        
        # Filter and reset index
        filtered_df = df[(~df['text_change'].str.contains('=>')) & (df['text_change'] != 'NoChange')].reset_index(drop=True)
    
        # Group and concatenate text changes
        grouped_df = filtered_df.groupby('id')['text_change'].apply(''.join).reset_index()
    
        # Find all occurrences of 'q+'
        grouped_df['text_change'] = grouped_df['text_change'].apply(lambda x: re.findall(r'q+', x))
    
        # Define a helper function to calculate various statistics
        def calculate_statistics(text_list):
            lengths = [len(i) for i in text_list]
            if not lengths:
                return [0] * 16
            return [
                len(text_list),  # word count
                np.mean(lengths),  # mean length
                np.max(lengths),  # max length
                np.std(lengths),  # std deviation
                np.min(lengths),  # min length
                np.median(lengths),  # median
                pd.Series(lengths).skew(),  # skewness
                pd.Series(lengths).kurtosis(),  # kurtosis
                gmean(lengths),  # geometric mean
                trim_mean(lengths, 0.1),  # trimmed mean
                entropy(lengths),  # entropy
                np.var(lengths),  # variance
                np.sum(lengths),  # sum
                np.prod(lengths),  # product
                np.sqrt(np.mean(lengths)),  # sqrt of mean
                pd.Series(lengths).cummax().iloc[-1],  # cumulative max
                pd.Series(lengths).cummin().iloc[-1],  # cumulative min
                pd.Series(lengths).cumsum().iloc[-1],  # cumulative sum
                pd.Series(lengths).cumprod().iloc[-1]  # cumulative product
            ]
    
        # Apply the helper function to each row
        stats_df = pd.DataFrame(grouped_df['text_change'].apply(calculate_statistics).tolist(), 
                                columns=['word_count', 'mean_length', 'max_length', 'std_dev', 
                                         'min_length', 'median_length', 'skewness', 'kurtosis',
                                         'geom_mean', 'trim_mean', 'entropy', 'variance',
                                         'sum_length', 'prod_length', 'sqrt_mean_length',
                                         'cummax_length', 'cummin_length', 'cumsum_length', 'cumprod_length'])
    
        # Compute additional derived features from the statistics (below)

        # Calculate the difference between max and min lengths
        stats_df['max_min'] = stats_df['max_length'] - stats_df['min_length']
        
        # Calculate the ratio of max length to min length (handling division by zero)
        stats_df['max_min_ratio'] = stats_df['max_length'] / stats_df['min_length'].replace(0, np.nan)
        
        # Calculate the ratio of max length to mean length
        stats_df['max_mean_ratio'] = stats_df['max_length'] / stats_df['mean_length']
        
        # Calculate the ratio of min length to mean length
        stats_df['min_mean_ratio'] = stats_df['min_length'] / stats_df['mean_length']
        
        # Calculate the ratio of max length to median length
        stats_df['max_median_ratio'] = stats_df['max_length'] / stats_df['median_length']
        
        # Calculate the ratio of min length to median length
        stats_df['min_median_ratio'] = stats_df['min_length'] / stats_df['median_length']
        
        # Calculate the ratio of max length to skewness (handling division by zero)
        stats_df['max_skew_ratio'] = stats_df['max_length'] / stats_df['skewness'].replace(0, np.nan)
        
        # Calculate the ratio of min length to skewness (handling division by zero)
        stats_df['min_skew_ratio'] = stats_df['min_length'] / stats_df['skewness'].replace(0, np.nan)
        
        # Calculate the ratio of max length to kurtosis (handling division by zero)
        stats_df['max_kurtosis_ratio'] = stats_df['max_length'] / stats_df['kurtosis'].replace(0, np.nan)
        
        # Calculate the ratio of min length to kurtosis (handling division by zero)
        stats_df['min_kurtosis_ratio'] = stats_df['min_length'] / stats_df['kurtosis'].replace(0, np.nan)
        
        # Calculate the ratio of max length to geometric mean
        stats_df['max_gmean_ratio'] = stats_df['max_length'] / stats_df['geom_mean']
        
        # Calculate the ratio of min length to geometric mean
        stats_df['min_gmean_ratio'] = stats_df['min_length'] / stats_df['geom_mean']
        
        # Calculate the ratio of max length to trimmed mean
        stats_df['max_trim_mean_ratio'] = stats_df['max_length'] / stats_df['trim_mean']
        
        # Calculate the ratio of min length to trimmed mean
        stats_df['min_trim_mean_ratio'] = stats_df['min_length'] / stats_df['trim_mean']
        
        # Calculate the ratio of max length to entropy (handling division by zero)
        stats_df['max_entropy_ratio'] = stats_df['max_length'] / stats_df['entropy'].replace(0, np.nan)
        
        # Calculate the ratio of min length to entropy (handling division by zero)
        stats_df['min_entropy_ratio'] = stats_df['min_length'] / stats_df['entropy'].replace(0, np.nan)
        
        # Calculate the ratio of max length to variance (handling division by zero)
        stats_df['max_var_ratio'] = stats_df['max_length'] / stats_df['variance'].replace(0, np.nan)
        
        # Calculate the ratio of min length to variance (handling division by zero)
        stats_df['min_var_ratio'] = stats_df['min_length'] / stats_df['variance'].replace(0, np.nan)
        
        # Calculate the ratio of max length to sum length
        stats_df['max_sum_ratio'] = stats_df['max_length'] / stats_df['sum_length']
        
        # Calculate the ratio of min length to sum length
        stats_df['min_sum_ratio'] = stats_df['min_length'] / stats_df['sum_length']
        
        # Calculate the ratio of max length to product length
        stats_df['max_prod_ratio'] = stats_df['max_length'] / stats_df['prod_length']
        
        # Calculate the ratio of min length to product length
        stats_df['min_prod_ratio'] = stats_df['min_length'] / stats_df['prod_length']
        
        # Calculate the ratio of max length to square root of mean length
        stats_df['max_sqrt_mean_ratio'] = stats_df['max_length'] / stats_df['sqrt_mean_length']
        
        # Calculate the ratio of min length to square root of mean length
        stats_df['min_sqrt_mean_ratio'] = stats_df['min_length'] / stats_df['sqrt_mean_length']
        
        # Calculate the ratio of max length to cumulative maximum length
        stats_df['max_cummax_ratio'] = stats_df['max_length'] / stats_df['cummax_length']
        
        # Calculate the ratio of min length to cumulative maximum length
        stats_df['min_cummax_ratio'] = stats_df['min_length'] / stats_df['cummax_length']
        
        # Calculate the ratio of max length to cumulative minimum length
        stats_df['max_cummin_ratio'] = stats_df['max_length'] / stats_df['cummin_length']
        
        # Calculate the ratio of min length to cumulative minimum length
        stats_df['min_cummin_ratio'] = stats_df['min_length'] / stats_df['cummin_length']
        
        # Calculate the ratio of max length to cumulative sum length
        stats_df['max_cumsum_ratio'] = stats_df['max_length'] / stats_df['cumsum_length']
        
        # Calculate the ratio of min length to cumulative sum length
        stats_df['min_cumsum_ratio'] = stats_df['min_length'] / stats_df['cumsum_length']
        
        # Calculate the ratio of max length to cumulative product length
        stats_df['max_cumprod_ratio'] = stats_df['max_length'] / stats_df['cumprod_length']
        
        # Calculate the ratio of min length to cumulative product length
        stats_df['min_cumprod_ratio'] = stats_df['min_length'] / stats_df['cumprod_length']
        
        # Calculate the ratio of max length to max_min (handling division by zero)
        stats_df['max_max_min_ratio'] = stats_df['max_length'] / stats_df['max_min'].replace(0, np.nan)
        
        # Calculate the ratio of min length to max_min (handling division by zero)
        stats_df['min_max_min_ratio'] = stats_df['min_length'] / stats_df['max_min'].replace(0, np.nan)


        # Concatenate the original ID column with the new statistics dataframe
        result_df = pd.concat([grouped_df[['id']], stats_df], axis=1)
    
        return result_df
    
    def make_features(self, dfx, name='Train Logs'):
        print(f'--- {name} Feature Engineering ---')
        
        # Create time features
        dfx, new_time_columns = create_time_features(dfx)
        self.discrete_numeric_features = list(set(self.discrete_numeric_features + new_time_columns))
        
        # Make a copy of the dataframe
        df = dfx.copy()
        
        new_df = pd.DataFrame()
        new_df['id'] = df['id'].unique()
        
        def make_agg_features(original_df, agg_functions, feature_names):
            tmp_df = pd.DataFrame()
            tmp_df['id'] = original_df['id'].unique()
            for dcf in feature_names:
                print(f'--- Making features from: {dcf} ---')
                
                # Group by 'id' and aggregate the features
                grouped = original_df.groupby('id')[dcf].agg(agg_functions)
                
                # Rename the columns
                grouped.columns = [f'{dcf}_{col}' for col in grouped.columns]
    
                # Merge the aggregated features into new_df
                tmp_df = pd.merge(tmp_df, grouped, left_on='id', right_index=True, how='left')
    
            return tmp_df
        
        # ------------------ Engineering shifted numerical features ------------------
        
        print("Engineering time data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']
            self.discrete_numeric_features.append(f'action_time_gap{gap}')
        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)
        
         # cursor position shift
        print("Engineering cursor position data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'cursor_position_shift{gap}'] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[f'cursor_position_shift{gap}']
            df[f'cursor_position_abs_change{gap}'] = np.abs(df[f'cursor_position_change{gap}'])
            self.discrete_numeric_features.append(f'cursor_position_change{gap}')
            self.discrete_numeric_features.append(f'cursor_position_abs_change{gap}')
        df.drop(columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)

        # word count shift
        print("Engineering word count data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'word_count_shift{gap}'] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[f'word_count_shift{gap}']
            df[f'word_count_abs_change{gap}'] = np.abs(df[f'word_count_change{gap}'])
            self.discrete_numeric_features.append(f'word_count_change{gap}')
            self.discrete_numeric_features.append(f'word_count_abs_change{gap}')
        df.drop(columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)
        
        # ------------------ Engineering aggregated features ------------------
        
        # Make features from numerical features
        print('\n=== Making features from numerical features ===')
        tmp_df_nf = make_agg_features(df, self.aggregation_functions, self.discrete_numeric_features)
        
        # Make features from categorical features
        print('\n=== Making features from categorical features ===')
        tmp_df_cf = make_agg_features(df, self.categorical_aggregations, self.categorical_features)
        
        # Merge the aggregated features into new_df
        new_df = pd.merge(new_df, tmp_df_nf, on='id', how='left')
        new_df = pd.merge(new_df, tmp_df_cf, on='id', how='left')
        
        feats = new_df
        print("\n=== Engineering punctuation counts data ===")
        tmp_df = self.match_punctuations(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        # Input words
        print("\n=== Engineering input words data ===")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')
         
        # TF-IDF features
        print("\n=== Engineering TF-IDF ===")
        tmp_df = make_text_features(df, name=name)
        feats = pd.merge(feats, tmp_df, on='id', how='left')
        
        # Create additional time features
        tmp_df, _ = create_additional_time_features(dfx)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        new_df = feats

        # ------------------ Engineering other features ------------------

        # Ratio-based features
        print("Engineering ratio-based features")
        feats['word_time_ratio'] = feats['word_count_max'] / feats['up_time_max']
        feats['word_event_ratio'] = feats['word_count_max'] / feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max']  / feats['up_time_max']
        
        # ------------------ Done feature engineering ------------------
        
        return new_df