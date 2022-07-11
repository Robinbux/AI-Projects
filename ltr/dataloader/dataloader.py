import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split

TRAIN_PATH = '../../data/training_set_VU_DM.csv'
TEST_PATH = '../../data/test_set_VU_DM.csv'
DATALOADER_DIR = Path(__file__).resolve().parent


class DataLoader:

    @staticmethod
    def load_train_test_dfs() -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_parquet(f'{DATALOADER_DIR}/parquets/train_df.parquet')
        test_df = pd.read_parquet(f'{DATALOADER_DIR}/parquets/test_df.parquet')
        return train_df, test_df

    @staticmethod
    def split_df_into_train_and_val_batches(train_df: pd.DataFrame, validation_size: float = 0.2,
                                            keep_search_id: bool = False) -> Tuple[
        npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        train_df_batches = list(dict(tuple(train_df.groupby('srch_id'))).values())
        if keep_search_id:
            train_df_batches = list(map(lambda df: df.drop('prop_id', axis=1), train_df_batches))
        else:
            train_df_batches = list(map(lambda df: df.drop('srch_id', axis=1).drop('prop_id', axis=1), train_df_batches))
        label_df_batches = [df.pop('relevance') for df in train_df_batches]
        # Convert to numpy
        np_train_df_batches = np.array([df.to_numpy() for df in train_df_batches], dtype="object")
        np_label_df_batches = np.array([df.to_numpy() for df in label_df_batches], dtype="object")
        if validation_size == 0:
            return np_train_df_batches, None, np_label_df_batches, None
        return train_test_split(np_train_df_batches, np_label_df_batches, test_size=validation_size, random_state=42)

    @staticmethod
    def random_train_valid_dfs(validation_size: float = 0.2) -> Tuple[
        npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        np_train_df_batches = np.load(f'{DATALOADER_DIR}/parquets/np_train_df_batches.npy', allow_pickle=True)
        np_label_df_batches = np.load(f'{DATALOADER_DIR}/parquets/np_label_df_batches.npy', allow_pickle=True)

        return train_test_split(np_train_df_batches, np_label_df_batches, test_size=validation_size, random_state=42)


def __create_standard_parquets() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df.to_parquet(f'{DATALOADER_DIR}/parquets/train_df.parquet')
    test_df.to_parquet(f'{DATALOADER_DIR}/parquets/test_df.parquet')


def __create_train_batches() -> None:
    pass
    # train_df = pd.read_parquet(f'{DATALOADER_DIR}/parquets/train_df.parquet')
    # train_df = Preprocessor.relevance(train_df)
    # train_df_batches = list(dict(tuple(train_df.groupby('srch_id'))).values())
    # train_df_batches = list(map(lambda df: df.drop('srch_id', axis=1), train_df_batches))
    # label_df_batches = [df.pop('relevance') for df in train_df_batches]
    # # Convert to numpy
    # np_train_df_batches = np.array([df.to_numpy() for df in train_df_batches], dtype="object")
    # np_label_df_batches = np.array([df.to_numpy() for df in label_df_batches], dtype="object")
    # # Save to files for quicker loading
    # np.save(f'{DATALOADER_DIR}/parquets/np_train_df_batches.npy', np_train_df_batches, allow_pickle=True)
    # np.save(f'{DATALOADER_DIR}/parquets/np_label_df_batches.npy', np_label_df_batches, allow_pickle=True)


if __name__ == '__main__':
    pass
    # __create_standard_parquets()
    #__create_train_batches()
