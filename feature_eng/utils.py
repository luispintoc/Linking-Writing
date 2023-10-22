import pandas as pd
import numpy as np
import re
import scipy
from scipy.fft import fft

## SELECT FROM HERE

# ------------- Check the size of a text change, 
# Note: Should be applied to column "text_change"

# Function for pandas apply that check number of large text changes
def count_large_text_changes(text_changes, size=20):
    return len([tc for tc in text_changes if len(tc) > size])

def count_extremely_large_text_changes(text_changes, size=100):
    return len([tc for tc in text_changes if len(tc) > size])

def count_tiny_text_changes(text_changes, size=5):
    return len([tc for tc in text_changes if len(tc) < size])

# ------------- For a given type of activity, count the number of times it occurs ------------ #
# Note: Should be applied to column "activity"

# Count nonproduction
def count_nonproduction(action_list):
    return len([action for action in action_list if action == 'Nonproduction'])

# count input
def count_input(action_list):
    return len([action for action in action_list if action == 'Input'])

# count remove/cut
def count_remove(action_list):
    return len([action for action in action_list if action == 'Remove/Cut'])

# Count Replace
def count_replace(action_list):
    return len([action for action in action_list if action == 'Replace'])

# Count Paste
def count_paste(action_list):
    return len([action for action in action_list if action == 'Paste'])

# ------------- For a given chunk of text that was moved, determine the size and the distance moved ------------ #
# Create move vectors features uses the four helpers below
def create_move_vectors_features(df): 
    
    # Create selection_vectors (the position of the selection before and after the move)
    df['selection_vectors'] = df['activity'].map(get_move_from_vectors)

    # Create functions from this
    df['distance_of_moved_selection'] = df['selection_vectors'].map(distance_of_move_of_selection)
    df['size_of_moved_selection'] = df['selection_vectors'].map(size_of_moved_selection)
    df.drop(['selection_vectors'], axis=1, inplace=True)

    return df

# Function to extract the distance vectors from the activity column
def split_activity(activity):
    return [np.array(a[1:-1].split(', '), dtype=int) for a in re.findall(r'\[[0-9]*, [0-9]*\]', activity)]

# Extract the vectors from the activity column when Move From is in the activity
def get_move_from_vectors(activity):
    if 'Move From' in activity:
        return split_activity(activity)
    else:
        return []
    
def distance_of_move_of_selection(selection_vectors):
    if len(selection_vectors) > 0:
        return selection_vectors[1][0] - selection_vectors[0][0]
    else:
        return 0

# How large was the selection
def size_of_moved_selection(selection_vectors):
    if len(selection_vectors) > 0:
        return selection_vectors[0][1] - selection_vectors[0][0]
    else:
        return 0

## Time features
### The total amount of time spent on the essay (as a fraction of total time allowed)
# The total amount of time the person spent writing the essay as a fraction of the total time
def fraction_of_time_spent_writing(writing_times):
    total_time = 1800000 # Half an hour in milliseconds
    max_time = max(writing_times)
    return max_time / total_time

# This function simply normalizes the text of Move From to be uniform, 
# should be used once Move From features have already been created
def normalize_move_from(activity):
    if 'Move From' in activity:
        return 'Move From'
    else:
        return activity
    
def create_action_time_features(df):
    # Normalize move from column
    df['activity'] = df['activity'].map(normalize_move_from)

    # Calculate average time, max time and total time of different actions
    action_time_features = df.groupby(['id', 'activity']).agg(
        {'action_time': ['mean', 'max', 'sum', 'count']}
    )

    # Flatten multi index columns
    action_time_features.columns = ['_'.join(col).strip() for col in action_time_features.columns.values]

    # Unstack multi index rows
    action_time_features = action_time_features.unstack('activity')

    # Re-flatten multi index columns
    action_time_features.columns = ['_'.join(col).strip() for col in action_time_features.columns.values]
    action_time_features.fillna(0, inplace=True) # Fill na with 0s 
    return action_time_features

def raw_aggregation_functions(df):

    # Create features related to individual action time
    action_time_features = create_action_time_features(df)

    # Create features related to moved selections of text
    df = create_move_vectors_features(df)

    # Feature engineering for typing behavior features
    typing_features = df.groupby('id').agg({
        'activity': 'count',                # Total number of activities
        'action_time': ['sum', 'mean'],     # Total and average action time
        'word_count': 'max',                # Maximum word count
        'text_change': 'nunique',           # Number of unique text changes
        'cursor_position': 'mean',           # Average cursor position
        'text_change' : count_large_text_changes,
        'text_change' : count_extremely_large_text_changes,
        'text_change' : count_tiny_text_changes,
        'activity': count_nonproduction,
        'activity': count_input,
        'activity': count_remove,
        'activity': count_replace,
        'activity': count_paste,
        'distance_of_moved_selection': ['mean', 'max'],
        'size_of_moved_selection': ['mean', 'max'],
        'up_time': fraction_of_time_spent_writing, # Amount of time spent on the essay,
    })

    # Flatten the multi-level column index
    typing_features.columns = ['_'.join(col).strip() for col in typing_features.columns.values]

    # Merge action time features with typing features
    features = pd.merge(typing_features, action_time_features, on='id')
    
    return features

# Optimize the function to calculate top N frequencies and their magnitudes for each 'id' using groupby and apply
def calculate_fft_features(group):

    group['pos'] = group['cursor_position']%30
    group['line'] = (group['cursor_position']/30).astype(int)

    # Perform Fourier Transform on 'pos'
    fft_values = fft(group['pos'])[1:]
    
    # Generate frequencies corresponding to the Fourier Transform values
    frequencies = np.fft.fftfreq(len(fft_values), 1)[1:]
    
    # Take absolute value to get magnitude
    fft_magnitude = np.abs(fft_values)
    
    # Identify indices where the frequencies are positive
    positive_indices = np.where(frequencies > 0)[0]
    
    # Filter out only positive frequencies and skip the zero frequency
    frequencies = frequencies[positive_indices]
    magnitudes = fft_magnitude[positive_indices]
    
    # Frequency Domain Features
    peak_freq = frequencies[np.argmax(magnitudes)]
    if np.sum(magnitudes) == 0:
        mean_freq = 0  # or some other appropriate default value
    else:
        mean_freq = np.average(frequencies, weights=magnitudes)

    median_freq = frequencies[len(magnitudes) // 2]
    bandwidth = np.ptp(frequencies)
    freq_skewness = scipy.stats.skew(magnitudes)
    freq_kurtosis = scipy.stats.kurtosis(magnitudes)

    # Other Features
    total_energy = np.sum(magnitudes ** 2)
    
    # Spectral Entropy
    psd_norm = np.abs(magnitudes) / np.sum(np.abs(magnitudes))
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + np.finfo(float).eps))
    
    # Spectral Flatness
    spectral_flatness = np.exp(np.mean(np.log(magnitudes + np.finfo(float).eps))) / np.mean(magnitudes)
    
    # Spectral Roll-off
    spectral_sum = np.cumsum(magnitudes)
    spectral_rolloff = frequencies[np.searchsorted(spectral_sum, 0.85 * spectral_sum[-1])]
    
    # Statistical Features
    mean_amplitude = np.mean(magnitudes)
    std_amplitude = np.std(magnitudes)
    skew_amplitude = scipy.stats.skew(magnitudes)
    kurtosis_amplitude = scipy.stats.kurtosis(magnitudes)

    features = {
        "Peak Frequency": peak_freq,
        "Mean Frequency": mean_freq,
        "Median Frequency": median_freq,
        "Bandwidth": bandwidth,
        "Frequency Skewness": freq_skewness,
        "Frequency Kurtosis": freq_kurtosis,
        "Total Energy": total_energy,
        "Spectral Entropy": spectral_entropy,
        "Spectral Flatness": spectral_flatness,
        "Spectral Roll-off": spectral_rolloff,
        "Mean Amplitude": mean_amplitude,
        "Std Amplitude": std_amplitude,
        "Skew Amplitude": skew_amplitude,
        "Kurtosis Amplitude": kurtosis_amplitude
    }
    
    return pd.Series(features)

def apply_fft_feats(df):
    return df.groupby('id').apply(calculate_fft_features)

##### MAKE SURE FUNC LIST IS UPDATED BEFORE RUNNING/PASTING TO NOTEBOOK #####
func_list = [raw_aggregation_functions, apply_fft_feats]


## END SELECTION HERE

def create_examples(func_list=func_list, input_path='../data', output_file=None):

    # Load the training data
    train_logs = pd.read_csv(f'{input_path}/train_logs.csv')

    # Create features
    features = pd.concat([func(train_logs) for func in func_list], axis=1)

    # Merge scores and return df
    train_scores = pd.read_csv(f'{input_path}/train_scores.csv')
    df = pd.merge(features, train_scores, on='id')

    # Save to file if output path is not None
    if output_file is not None:
        # Is output path a string
        if isinstance(output_file, str):
            df.to_csv(output_file, index=False)
        else:
            print('Output path must be a string, not saving to file')
    return df