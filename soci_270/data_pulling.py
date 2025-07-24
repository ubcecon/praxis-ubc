import kaggle
import pandas as pd
import os
import glob
import subprocess
import shutil
import ast
from dateutil.parser import parse
from dateutil.tz import gettz


DATASETS_KAGGLE = {'disinfo_ira': {'id': 'fivethirtyeight/russian-troll-tweets', 'path': 'data/russian_troll_tweets'},
                   'control_en': {'id': 'kazanova/sentiment140', 'path': 'data/sentiment140_dataset'},
                   'disinfo_war': {'id': 'dariusalexandru/russian-propaganda-tweets-vs-western-tweets-war', 'path': 'data/russian_propaganda_vs_western'}}


def download_kaggle_datasets(datasets_config):
    """Download all specified datasets from Kaggle"""
    for key, info in datasets_config.items():
        if not (os.path.exists(info['path']) and os.listdir(info['path'])):
            kaggle.api.dataset_download_files(info['id'], path=info['path'], unzip=True)
        else:
            print('skipping')

def download_github_repo(repo_url, clone_path):
    """Clones dataset repository from GitHub"""
    if not (os.path.exists(clone_path) and os.path.isdir(os.path.join(clone_path, '.git'))):
        subprocess.run(['git', 'clone', repo_url, clone_path], check=True, capture_output=True)
    else:
        print('skipping, data there already')


def standardize_dataframe(df, source_name):
    """Standardizes df by parsing times/dates and converting to UTC"""
    dt_series = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns, UTC]')

    if source_name == 'war_propaganda':
        naive_dt = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        offset = pd.to_timedelta(df['timezone'], unit='m', errors='coerce')
        dt_series = (naive_dt - offset).dt.tz_localize('UTC')
    else:
        datetime_col_map = {'ira': 'publish_date', 'sentiment140': 'date'}
        timestamp_col = datetime_col_map.get(source_name)

        if timestamp_col and timestamp_col in df.columns:
            if source_name == 'sentiment140':
                date_str_series = df[timestamp_col].str.replace(' PDT ', ' -0700 ', regex=False)
                aware_dt = pd.to_datetime(date_str_series, 
                    format='%a %b %d %H:%M:%S %z %Y', 
                    errors='coerce')
                dt_series = aware_dt.dt.tz_convert('UTC')

            else: 
                naive_dt = pd.to_datetime(df[timestamp_col], errors='coerce')
                dt_series = naive_dt.dt.tz_localize('UTC')

    df['date'] = dt_series.dt.date
    df['time'] = dt_series.dt.time

    username_col_map = {'ira': 'author', 'war_propaganda': 'username', 'sentiment140': 'user'}
    author_col = username_col_map.get(source_name)
    if author_col and author_col in df.columns:
        df['username'] = df[author_col]
    print('fixed dates')
    return df


def load_ira_data(path):
    """Loads and processes the IRA disinformation dataset"""
    disinfo_files = glob.glob(os.path.join(path, 'IRAhandle_tweets_*.csv'))
    df = pd.concat((pd.read_csv(f, low_memory=False, encoding='utf-8-sig') for f in disinfo_files), ignore_index=True)
    df = standardize_dataframe(df, 'ira')
    df['source'] = 'IRA'
    df['is_propaganda'] = 1
    return df

def load_war_propaganda_data(path):
    """loads and processes the 2022 War Propaganda dataset"""
    df = pd.read_csv(path, encoding='utf-16', delimiter='\t')
    df.rename(columns={'tweet': 'content'}, inplace=True)
    json_columns = ['mentions', 'urls', 'photos', 'hashtags', 'cashtags', 'reply_to', 'geo', 'place']
    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith(('[', '{')) else x)

    df = standardize_dataframe(df, 'war_propaganda')
    df['source'] = 'WarPropaganda'
    df['is_propaganda'] = 1
    return df

def load_control_en_data(path):
    """Loads and processes sentiment140"""
    sentiment_colnames = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(path, encoding='latin-1', header=None, names=sentiment_colnames)
    df.rename(columns={'text': 'content'}, inplace=True)
    df = standardize_dataframe(df, 'sentiment140')
    df['source'] = 'Sentiment140'
    df['is_propaganda'] = 0
    df['language'] = 'english'
    return df

def load_control_ru_data(path):
    """Loads and processes rusentitweet"""
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.rename(columns={'text': 'content'}, inplace=True)
    df = standardize_dataframe(df, 'rusentitweet')
    df['source'] = 'RuSentitweet'
    df['is_propaganda'] = 0
    df['language'] = 'russian'
    return df

def align_schemas(dfs_list, master_columns):
    aligned_dfs = []
    for df in dfs_list:
        aligned_df = df.reindex(columns=master_columns)
        aligned_dfs.append(aligned_df)
    return aligned_dfs

def create_advanced_disinformation_dataset():
    os.makedirs('data', exist_ok=True)

    kaggle.api.authenticate()

    download_kaggle_datasets(DATASETS_KAGGLE)
    print('downloaded kaggle datasets')
    download_github_repo('https://github.com/sismetanin/rusentitweet.git', 'data/rusentitweet')
    print('downloaded github repo')
    df_ira = load_ira_data(DATASETS_KAGGLE['disinfo_ira']['path'])
    master_columns = df_ira.columns.tolist()

    df_war_propaganda = load_war_propaganda_data(
        os.path.join(DATASETS_KAGGLE['disinfo_war']['path'], 'russian_propaganda_tweets.csv'))
    df_control_en = load_control_en_data(
        os.path.join(DATASETS_KAGGLE['control_en']['path'], 'training.1600000.processed.noemoticon.csv'))
    df_control_ru = load_control_ru_data('data\\rusentitweet\\rusentitweet_full.csv')
    all_dfs = [df_ira, df_war_propaganda, df_control_en, df_control_ru]
    aligned_dfs = align_schemas(all_dfs, master_columns)
    df_combined = pd.concat(aligned_dfs, ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    output_filename = 'soci_270\data\combined_disinformation_dataset_final.csv'
    df_combined.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print('done')

create_advanced_disinformation_dataset()