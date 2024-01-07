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


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


manual_list = [
"median_lantency-latencies",
"intro_ratio-paragraph_ratios",
"total_word_count-paragraph_ratios",
"word_count_body-paragraph_ratios",
"relative_body_size-paragraph_ratios",
"relative_intro_size-paragraph_ratios",
"body_end-paragraph_ratios",
"product_len-paragraph_ratios",
"essay_len-paragraph_ratios",
"process_variance-paussed_features",
"count_dist_per_event-cursor_visits",
"count_line_changes-cursor_visits",
"count_time_per_line_bucket_0/[3]-cursor_visits",
"count_time_per_line_bucket_1/[3]-cursor_visits",
"count_time_per_line_bucket_0/[5]-cursor_visits",
"count_time_per_line_bucket_1/[5]-cursor_visits",
"count_time_per_line_bucket_2/[5]-cursor_visits",
"count_time_per_line_bucket_1/[10]-cursor_visits",
"count_time_per_line_bucket_2/[10]-cursor_visits",
"count_time_per_line_bucket_3/[10]-cursor_visits",
"count_time_per_line_bucket_4/[10]-cursor_visits",
"count_time_per_line_bucket_5/[10]-cursor_visits",
"count_time_per_line_bucket_6/[10]-cursor_visits",
"count_time_per_line_bucket_7/[10]-cursor_visits",
"sum_time_per_line_bucket_0/[3]-cursor_visits",
"sum_time_per_line_bucket_1/[3]-cursor_visits",
"sum_time_per_line_bucket_2/[3]-cursor_visits",
"sum_time_per_line_bucket_2/[10]-cursor_visits",
"sum_time_per_line_bucket_3/[10]-cursor_visits",
"sum_time_per_line_bucket_6/[10]-cursor_visits",
"sum_time_per_line_bucket_5/[10]-cursor_visits",
"skew_time_per_line_bucket_1/[5]-cursor_visits",
"skew_time_per_line_bucket_0/[10]-cursor_visits",
"skew_time_per_line_bucket_5/[10]-cursor_visits",
"activity_0_cnt-count_bursts",
"text_change_0_cnt-count_bursts",
"text_change_5_cnt-count_bursts",
"text_change_6_cnt-count_bursts",
"text_change_7_cnt-count_bursts",
"text_change_8_cnt-count_bursts",
"text_change_11_cnt-count_bursts",
"down_event_0_cnt-count_bursts",
"down_event_6_cnt-count_bursts",
"down_event_8_cnt-count_bursts",
"down_event_13_cnt-count_bursts",
"down_event_14_cnt-count_bursts",
"up_event_0_cnt-count_bursts",
"input_word_count-count_bursts",
"action_time_sum-count_bursts",
"cursor_position_mean-count_bursts",
"word_count_mean-count_bursts",
"cursor_position_std-count_bursts",
"word_count_std-count_bursts",
"word_count_median-count_bursts",
"cursor_position_max-count_bursts",
"word_count_max-count_bursts",
"word_count_quantile-count_bursts",
"inter_key_median_lantency-count_bursts",
"mean_pause_time-count_bursts",
"P-bursts_median-count_bursts",
"word_len_count-word_sent_parag_agg",
"word_len_min-word_sent_parag_agg",
"word_len_median-word_sent_parag_agg",
"word_len_sum-word_sent_parag_agg",
"sent_len_last-word_sent_parag_agg",
"sent_len_sum-word_sent_parag_agg",
"sent_word_count_sum-word_sent_parag_agg",
"paragraph_len_q1-word_sent_parag_agg",
"paragraph_len_median-word_sent_parag_agg",
"paragraph_len_sum-word_sent_parag_agg",
"essay_len-word_sent_parag_agg",
"duration_mean-IWD",
"duration_count-IWD",
"duration_q1-IWD",
"duration_q3-IWD",
"intraword pause_count-IWD",
"intraword pause_mean-IWD",
"intraword pause_max-IWD",
"intraword pause_q1-IWD",
"intraword pause_median-IWD",
"intraword pause_q3-IWD",
"intraword pause_std-IWD",
"IWD_count-IWD",
"IWD_q3-IWD",
"insert_q-avg_char_insert_per_minute",
"all_event_cnt-key_mouse",
"min-IKI_word",
"Total Energy-fft",
"Spectral Entropy-fft",
"Std Amplitude-fft",
"Mean Amplitude-fft",
"pauses_3_sec-latencies",
"!-punctuation",
"'-punctuation",
"(-punctuation",
",-punctuation",
"--punctuation",
"/-punctuation",
"=-punctuation",
">-punctuation",
"ArrowLeft-punctuation",
"ArrowDown-punctuation",
"CapsLock-punctuation",
"Delete-punctuation",
"End-punctuation",
"Middleclick-punctuation",
"NumLock-punctuation",
"Rightclick-punctuation",
"Shift-punctuation",
"q-punctuation",
"x-punctuation",
"}-punctuation",
"Total-punctuation",
]