import pandas as pd
from scipy.stats import ks_2samp


def data_drift_report(df_a, df_b, name_a, name_b):

    print('Proportion of words, punctuation marks, digits, \
    symbols, capitalized words, and entities in documents.')
    print('\n')
    df_var_rate = compare_vars_per_token(df_a, df_b, name_a, name_b)
    print(df_var_rate)
    print('\n')

    print('Document length:\n')
    df_var_mean = compare_mean_var_per_doc(df_a, df_b, name_a, name_b)
    print(df_var_mean)
    print('\n')

    print('POS tag distributions:\n')
    df_pos = comparision_pos(df_a, df_b, name_a, name_b)
    print(df_pos)
    print('\n')

    print('Kolmogorov-Smirnov tests for distributions:\n\
    - token-per-sentence \n\
    - entity-per-sentence\n')
    ks_stats_df = ks_stats_var_per_sent(df_a, df_b)
    print(ks_stats_df)
    print('\n')


def compare_vars_per_token(df_a, df_b, name_a, name_b):
    df_ratio = pd.DataFrame()
    df_ratio = vars_per_token(df_a, df_ratio, name_a)
    df_ratio = vars_per_token(df_b, df_ratio, name_b)
    return df_ratio


def vars_per_token(df, df_sum, set_name):
    vars = [
    'words',
    'punctuation',
    'digits',
    'symbols',
    'capital',
    'entities'
    ]
    var_sum = df[vars].sum()
    tokens_sum = df['tokens'].sum()
    df_sum[set_name] = round(var_sum / tokens_sum, 2)
    return df_sum


def compare_mean_var_per_doc(df_a, df_b, name_a, name_b):
    df_ratio = pd.DataFrame()
    df_ratio = mean_var_per_doc(df_a, df_ratio, name_a)
    df_ratio = mean_var_per_doc(df_b, df_ratio, name_b)
    return df_ratio


def mean_var_per_doc(df, df_sum, set_name):
    vars = [
        'tokens',
        'words',
        'sentences',
        'entities',
        ]
    var_mean = df[vars].mean().astype(int)
    df_sum[set_name] = var_mean
    return df_sum


def comparision_pos(df_a, df_b, name_a, name_b):
    """
    POS tag distributions
    """

    pos_sum = pd.DataFrame()
    pos_a = df_a['pos_distrib'].apply(pd.Series).fillna(0)
    pos_b = df_b['pos_distrib'].apply(pd.Series).fillna(0)

    pos_sum[name_a] = pos_a.mean().round(2)
    pos_sum[name_b] = pos_b.mean().round(2)
    return pos_sum


def ks_stats_var_per_sent(df_a, df_b):
    """
    Kolmogorov-Smirnov tests for token-per-sentence
    and entity-per-sentence distributions.
    """

    vars = ['tokens_per_sent', 'entities_per_sent']
    ks_stats = []

    for var in vars:
        var_list_a = df_a[var].explode(ignore_index=True).to_list()
        var_list_b = df_b[var].explode(ignore_index=True).to_list()

        stat, p_value = ks_2samp(var_list_a, var_list_b)
        ks_stats.append({
            'var': var,
            'KS statistic': stat,
            'p-value': p_value
            })

    ks_stats_df = pd.DataFrame(ks_stats)
    ks_stats_df.set_index('var', inplace=True)
    return ks_stats_df
