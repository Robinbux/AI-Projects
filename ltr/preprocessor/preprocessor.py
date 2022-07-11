import pandas as pd
import numpy as np
import datetime
from typing import List
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src_clean.dataloader import DataLoader
from src_clean.utils.utils import add_suffix

TRAIN_PATH = '../../data/training_set_VU_DM.csv'
TEST_PATH = '../../data/test_set_VU_DM.csv'

STATISTICS_FEATURES = [
    #'visitor_hist_starrating',
    #'prop_log_historical_price',
    #'srch_children_count',
    #'visitor_hist_adr_usd',
    'price_usd', ##
    #'srch_room_count',
    'prop_starrating', ##
    #'orig_destination_distance',
    #'srch_destination_id',
    'prop_review_score', ##
    #'srch_saturday_night_bool',
    #'prop_brand_bool',
    #'srch_length_of_stay',
    #'srch_booking_window',
    'prop_location_score1', ##
    #'srch_query_affinity_score',
    #'promotion_flag',
    'prop_location_score2', ##
    #'srch_adults_count',
    #'random_bool',
]

COLS_TO_DROP = [
    'srch_id',
    'site_id',
    'prop_id',
    'visitor_location_country_id',
    'prop_country_id',
    'gross_bookings_usd',
]


class Preprocessor:

    @staticmethod
    def create_parquets(drop_search: bool = True) -> None:
        train_df = Preprocessor.preprocess(pd.read_csv(TRAIN_PATH), drop_search=drop_search)
        test_df = Preprocessor.preprocess(pd.read_csv(TEST_PATH), drop_search=False)
        if drop_search:
            train_df.to_parquet('train_df_preprocessed.parquet')
        else:
            train_df.to_parquet('train_df_with_srch_id_preprocessed.parquet')
        test_df.to_parquet('test_df_preprocessed.parquet')

    @staticmethod
    def preprocess(df: pd.DataFrame, exclude_cols_to_drop: List[str] = [],
                   set_average_position: bool = False) -> pd.DataFrame:
        # df = Preprocessor.__prepro_comp_columns(df)
        #df = Preprocessor.__prepro_datetime(df)
        # df = Preprocessor.__prepro_add_extra_features(df)
        df = Preprocessor.__prepro_drop_cols(df, exclude_cols_to_drop=exclude_cols_to_drop)
        #df = Preprocessor.__prepro_imput(df)

        return df

    @staticmethod
    def relevance(df: pd.DataFrame, remove_irrelevant: bool = False) -> pd.DataFrame:
        # Include position here?
        df['relevance'] = (df['booking_bool'] * 4 + df['click_bool']).round().astype('Int64')
        df.drop(['booking_bool', 'click_bool'], axis=1, inplace=True)
        return df

    @staticmethod
    def sklearn_prepro(df, cols_to_exclude=None):
        if cols_to_exclude is None:
            cols_to_exclude = []
        df.loc[:, ~df.columns.isin(cols_to_exclude)] = MinMaxScaler().fit_transform(
            df.loc[:, ~df.columns.isin(cols_to_exclude)])
        return df

    @staticmethod
    def impute_position(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
        mean_pos_df = train_df[train_df['random_bool'] == 0][['prop_id', 'position']].groupby('prop_id').mean()
        result_df = df.merge(mean_pos_df, on='prop_id', how='left')
        return result_df

    @staticmethod
    def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
        copy_df = df.copy()
        copy_df = copy_df[['prop_id'] + STATISTICS_FEATURES]

        # Create statistics df's
        df_mean = copy_df.groupby('prop_id', as_index=False).mean().fillna(-1)
        df_mean = add_suffix(df_mean, suffix='_mean', cols_to_exclude=['prop_id'])
        df_std = copy_df.groupby('prop_id', as_index=False).std().fillna(-1)
        df_std = add_suffix(df_std, suffix='_std', cols_to_exclude=['prop_id'])
        df_median = copy_df.groupby('prop_id', as_index=False).median().fillna(-1)
        df_median = add_suffix(df_median, suffix='_median', cols_to_exclude=['prop_id'])

        # Merge on prop_id
        df = df.merge(df_mean, on='prop_id')
        df = df.merge(df_std, on='prop_id')
        df = df.merge(df_median, on='prop_id')

        return df

    @staticmethod
    def __prepro_comp_columns(df: pd.DataFrame) -> pd.DataFrame:
        # Drop all other comp columns, except for "rate"
        df = df.drop(df.filter(regex='(comp._inv|comp._rate.+)').columns, axis=1)

        # Create new col
        df['comp_cheaper'] = 0
        df['comp_cheaper'] = df['comp_cheaper'].mask((df.filter(regex='comp') > 0).any(axis=1), 1).mask(
            (df.filter(regex='comp') < 0).any(axis=1), -1)
        # Drop old comp cols
        return df.drop(df.filter(regex='comp._rate').columns, axis=1)

    @staticmethod
    def __date_to_season(date: datetime.datetime) -> int:
        if 3 <= date.month < 6:
            return 0
        elif 6 <= date.month < 9:
            return 1
        elif 9 <= date.month < 12:
            return 2
        return 3

    @staticmethod
    def __prepro_datetime(df: pd.DataFrame) -> pd.DataFrame:
        df['date_time'] = pd.to_datetime(df['date_time'], format='%Y-%m-%d')
        df['season'] = (df['date_time'] + pd.to_timedelta(df['srch_booking_window'], unit='D')).apply(
            Preprocessor.__date_to_season)
        # Drop old date_time columns
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder_df = pd.DataFrame(encoder.fit_transform(df[['season']]).toarray())
        df = df.join(encoder_df)
        df.drop(['date_time', 'season'], axis=1, inplace=True)
        return df

    @staticmethod
    def __prepro_drop_cols(df: pd.DataFrame, exclude_cols_to_drop: List[str] = []) -> pd.DataFrame:
        cols_to_drop_copy = COLS_TO_DROP.copy()
        for col in exclude_cols_to_drop:
            cols_to_drop_copy.remove(col)
        df.drop(cols_to_drop_copy, axis=1, errors='ignore', inplace=True)
        df.drop(df.filter(regex='comp').columns, axis=1, inplace=True)
        return df

    @staticmethod
    def __prepro_imput(df: pd.DataFrame) -> pd.DataFrame:
        df['prop_review_score'] = df['prop_review_score'].fillna(df['prop_review_score'].mean())
        df['prop_location_score2'] = df['prop_location_score2'].fillna(0)
        return df

    @staticmethod
    def __prepro_add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
        # Adding new variables
        df['mean_price'] = df['price_usd'].groupby(df['prop_id']).transform('mean')
        df['prop_price_diff'] = np.exp(df['prop_log_historical_price']) - df['price_usd']
        # df['vist_price_diff'] = df['visitor_hist_adr_usd'] - df['price_usd']
        # interpolate missing values
        # df['vist_price_diff'] = df.groupby('prop_country_id')['vist_price_diff'].apply(lambda x : x.interpolate(method = "spline", order = 1, limit_direction = "both"))
        # df['vist_price_diff'] = df.groupby('prop_country_id')['vist_price_diff'].apply(lambda x: x.fillna(x.mean()))
        # df.value = df.value.fillna(df.value.mean())

        df['fee_person'] = (df['srch_room_count'] * df['price_usd']) / (
                df['srch_adults_count'] + df['srch_children_count'])
        df['total_fee'] = df['srch_room_count'] * df['price_usd']

        # df.drop(['prop_log_historical_price'], axis=1, errors='ignore', inplace=True)

        return df


if __name__ == "__main__":
    # Preprocessor.create_parquets()
    train_df, test_df = DataLoader.load_train_test_dfs()
    stat_df = Preprocessor.calculate_statistics(train_df)
    print(stat_df)
    # train_df = Preprocessor.preprocess(train_df, exclude_cols_to_drop=['srch_id', 'prop_id'])
