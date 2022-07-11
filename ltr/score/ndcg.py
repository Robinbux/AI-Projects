import sys

import pandas as pd
from sklearn.metrics import ndcg_score
import numpy as np
from multiprocessing import Pool
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

NDCG_DIR = Path(__file__).resolve().parent

def relevance(df: pd.DataFrame) -> pd.DataFrame:
    df['relevance'] = df['booking_bool'] * 4 + df['click_bool']
    df.drop(['booking_bool', 'click_bool'], axis=1, inplace=True)
    return df


def custom_ndcg(a, b, k=5):
    """Custom funciton for kword arg"""
    return ndcg_score(a, b, k=k)


def create_parquet() -> None:
    recon_df = pd.read_csv('testset_reconstructed.csv')
    recon_df = relevance(recon_df)
    recon_df = recon_df.sort_values(by=['srch_id'])
    recon_df.to_parquet('recon_df.parquet')


def calculate_ndcg(custom_rank_path: str) -> float:
    custom_rank_df = pd.read_csv(custom_rank_path)
    custom_rank_df['dummy_relevance'] = list(range(len(custom_rank_df), 0, -1))
    recon_df = pd.read_parquet(f'{NDCG_DIR}/recon_df.parquet', columns=['srch_id', 'prop_id', 'relevance'])
    recon_df = recon_df.sort_values(by=['srch_id', 'prop_id'])
    custom_rank_df = custom_rank_df.sort_values(by=['srch_id', 'prop_id'])
    recon_tuples = tuple(recon_df.groupby('srch_id'))
    custom_rank_tuples = tuple(custom_rank_df.groupby('srch_id'))
    true_rels = list([[list(recon_tup[1]['relevance'])] for recon_tup in recon_tuples])
    dummy_rels = list([[list(custom_rank_tup[1]['dummy_relevance'])] for custom_rank_tup in custom_rank_tuples])
    zipped_list = list(zip(true_rels, dummy_rels))
    pool = Pool()
    score = pool.starmap(custom_ndcg,zipped_list)
    return np.average(score)


#PATH = '/Users/robinbux/Desktop/VU/Period5/DataMining/DMT_2022/Assignment_2/src/results/result.csv'
#PATH = '/Users/robinbux/Desktop/VU/Period5/DataMining/expedia2013/results/submission.csv'
if __name__ == "__main__":
    # create_parquet()
    ndcg = calculate_ndcg(sys.argv[1])
    print(f"NDCG Score: {ndcg}")
