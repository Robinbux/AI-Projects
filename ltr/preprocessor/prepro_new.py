from datetime import datetime

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import zscore

from src_clean.dataloader import DataLoader
from src_clean.utils.utils import add_suffix

import warnings

warnings.filterwarnings('ignore')

STATISTICS_FEATURES = [
    'visitor_hist_starrating',
    'prop_log_historical_price',
    'srch_children_count',
    'visitor_hist_adr_usd',
    'price_usd',  ##
    'srch_room_count',
    'prop_starrating',  ##
    'orig_destination_distance',
    'prop_review_score',  ##
    'srch_saturday_night_bool',
    'prop_brand_bool',
    'srch_length_of_stay',
    'srch_booking_window',
    'prop_location_score1',  ##
    'srch_query_affinity_score',
    'promotion_flag',
    'prop_location_score2',  ##
    'srch_adults_count',
    'random_bool',
]

COLS_TO_DROP = [
    'srch_id',
    'site_id',
    'prop_id',
    'visitor_location_country_id',
    'prop_country_id',
    'gross_bookings_usd',
    'date_time',
    'srch_destination_id',
    'booking_bool',
    'click_bool'
]

FEATURE_NORMALIZATION_COLS = [
    'price_usd',
    'prop_log_historical_price',
    'prop_starrating',  ##
    'prop_review_score',  ##
    'prop_brand_bool',
    'prop_location_score1',  ##
    'prop_location_score2',  ##
    'price_diff_per_prop_id',
    'price_diff_per_srch_id',
    'visitor_price_diff',
    'visitor_starrating_diff'
]

FEATURE_NORMALIZATION_GROUP_COLS = [
    'srch_id',
    'prop_id',
    'srch_destination_id'
]


class PreprocessorTwo:

    @staticmethod
    def relevance(df: pd.DataFrame) -> pd.DataFrame:
        df['relevance'] = (df['booking_bool'] * 4 + df['click_bool']).round().astype('Int64')
        # df.drop(['booking_bool', 'click_bool'], axis=1, inplace=True)
        return df

    @staticmethod
    def calculate_statistics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Use combined_df for test and ONLY train for train
        orig_train_df = train_df.copy()
        orig_test_df = test_df.copy()
        for feature in FEATURE_NORMALIZATION_GROUP_COLS:
            if feature == 'srch_id': continue
            for idx, df in enumerate([train_df, test_df]):
                if df is test_df:
                    copy_train_df = orig_train_df[[feature] + STATISTICS_FEATURES]
                    copt_test_df = orig_test_df[[feature] + STATISTICS_FEATURES]
                    stat_df = pd.concat([copy_train_df, copt_test_df])
                else:
                    stat_df = orig_train_df[[feature] + STATISTICS_FEATURES]
                # Create statistics df's
                df_mean = stat_df.groupby(feature, as_index=False).mean().fillna(-1)
                df_mean = add_suffix(df_mean, suffix=f'_mean_on_{feature}', cols_to_exclude=[feature])
                df_std = stat_df.groupby(feature, as_index=False).std().fillna(-1)
                df_std = add_suffix(df_std, suffix=f'_std_on_{feature}', cols_to_exclude=[feature])
                df_median = stat_df.groupby(feature, as_index=False).median().fillna(-1)
                df_median = add_suffix(df_median, suffix=f'_median_on_{feature}', cols_to_exclude=[feature])
                df = df.reset_index().merge(df_mean, how="left").set_index('index')
                df = df.reset_index().merge(df_std, how="left").set_index('index')
                df = df.reset_index().merge(df_median, how="left").set_index('index')
                if idx == 0:
                    train_df = df
                else:
                    test_df = df
        return train_df, test_df

    @staticmethod
    def calculate_statistics_srch_id(df: pd.DataFrame) -> pd.DataFrame:
        feature = 'srch_id'
        copy_df = df.copy()[[feature] + STATISTICS_FEATURES]
        df_mean = copy_df.groupby(feature, as_index=False).mean().fillna(-1)
        df_mean = add_suffix(df_mean, suffix=f'_mean_on_{feature}', cols_to_exclude=[feature])
        df_std = copy_df.groupby(feature, as_index=False).std().fillna(-1)
        df_std = add_suffix(df_std, suffix=f'_std_on_{feature}', cols_to_exclude=[feature])
        df_median = copy_df.groupby(feature, as_index=False).median().fillna(-1)
        df_median = add_suffix(df_median, suffix=f'_median_on_{feature}', cols_to_exclude=[feature])
        df = df.reset_index().merge(df_mean, how="left").set_index('index')
        df = df.reset_index().merge(df_std, how="left").set_index('index')
        df = df.reset_index().merge(df_median, how="left").set_index('index')
        return df

    @staticmethod
    def preprocess(df: pd.DataFrame, exclude_cols_to_drop: List[str] = []) -> pd.DataFrame:
        df = PreprocessorTwo.drop_cols(df, exclude_cols_to_drop=exclude_cols_to_drop)
        return df

    @staticmethod
    def prepro_comp_columns(df: pd.DataFrame) -> pd.DataFrame:
        # Drop all other comp except 5 and 7
        comp_cols_to_drop = [col for col in df.columns if 'comp' in col]
        comp_cols_to_drop.remove('comp2_rate_percent_diff')
        comp_cols_to_drop.remove('comp7_rate_percent_diff')
        comp_cols_to_drop.remove('comp5_rate')
        comp_cols_to_drop.remove('comp8_rate')
        comp_cols_to_drop.remove('comp2_rate')
        df = df.drop(comp_cols_to_drop, axis=1)
        return df

    @staticmethod
    def visitor_starrating_diff(df: pd.DataFrame) -> pd.DataFrame:
        df['visitor_starrating_diff'] = df['visitor_hist_starrating'] - df['prop_starrating']
        return df

    @staticmethod
    def visitor_price_diff(df: pd.DataFrame) -> pd.DataFrame:
        df['visitor_price_diff'] = df['visitor_hist_adr_usd'] - df['price_usd']
        return df

    @staticmethod
    def add_price_difference_srch_id(df: pd.DataFrame) -> pd.DataFrame:
        mean_price_per_srchid = df.groupby(["srch_id"])['price_usd'].mean().reset_index(name='price_usd_mean_per_srch_id')
        df = df.merge(mean_price_per_srchid, on=['srch_id'], how='left')
        df['price_diff_per_srch_id'] = df['price_usd'] - df['price_usd_mean_per_srch_id']
        df.drop('price_usd_mean_per_srch_id', axis=1, inplace=True)
        return df

    @staticmethod
    def add_price_difference_prop_id(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        combined_df = pd.concat([train_df[['prop_id', 'price_usd']], test_df[['prop_id', 'price_usd']]])
        df_mean = combined_df.groupby(['prop_id'])['price_usd'].mean().reset_index(name='price_usd_mean_per_prop_id')

        tt = test_df.reset_index().merge(df_mean, on=['prop_id'], how='left')
        tt['price_diff_per_prop_id'] = tt['price_usd'] - tt['price_usd_mean_per_prop_id']
        tt.drop(['price_usd_mean_per_prop_id', 'index'], axis=1, inplace=True)

        df_mean = train_df.groupby(['prop_id'])['price_usd'].mean().reset_index(name='price_usd_mean_per_prop_id')
        tr = train_df.reset_index().merge(df_mean, on=['prop_id'], how='left')
        tr['price_diff_per_prop_id'] = tr['price_usd'] - tr['price_usd_mean_per_prop_id']
        tr.drop(['price_usd_mean_per_prop_id', 'index'], axis=1, inplace=True)

        return tr, tt

    @staticmethod
    def add_price_order(df: pd.DataFrame) -> pd.DataFrame:
        df["price_rank"] = df.groupby("srch_id", as_index=False)["price_usd"].rank("dense", ascending=True)
        df['price_rank'] = df['price_rank']/df.groupby(["srch_id"])['price_usd'].transform('count')
        return df

    @staticmethod
    def feature_normalization(df: pd.DataFrame) -> pd.DataFrame:
        for feature in FEATURE_NORMALIZATION_COLS:
            for group_by_feature in FEATURE_NORMALIZATION_GROUP_COLS:
                cols = ['srch_id', 'prop_id', feature]
                if group_by_feature != 'srch_id' and group_by_feature != 'prop_id':
                    cols.append(group_by_feature)
                feature_df = df[cols]
                feature_df[feature] = np.log10(feature_df[feature])
                groups = feature_df[[group_by_feature, feature]].groupby(group_by_feature)
                mean, std = groups.transform("mean"), groups.transform("std")
                feature_df[feature] = (feature_df[mean.columns] - mean) / std
                feature_df.rename(columns={feature: f'{feature}_log10normalized_by_{group_by_feature}'}, inplace=True)
                df = df.reset_index().merge(feature_df, how='left').set_index('index')
        return df

    # @staticmethod
    # def remove_outliers(df: pd.DataFrame, feature="price_usd") -> pd.DataFrame:
    #     df_copy = df.copy()
    #
    #     Q1= df_copy[feature].quantile(0.15)
    #     Q3 = df_copy[feature].quantile(0.85)
    #     IQR = Q3 - Q1
    #     upper_limit = Q3 + 1.5 * IQR
    #     lower_limit = Q1 - 1.5 * IQR
    #
    #     print("Removed Outliers: ",  df_copy.loc[~df_copy[feature].between(lower_limit,
    #                                                                        upper_limit, inclusive=False), feature])
    #     pos_df = df_copy[['prop_id', 'price_usd']]
    #     df_mean = pos_df.groupby('prop_id', as_index=False).mean()
    #     df_copy.loc[df_copy[feature]>=lower_limit and df_copy[feature]<= upper_limit]
    #     return df_copy

    @staticmethod
    def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['price_usd'] <= 10_000]
        return df

    @staticmethod
    def mean_and_std_pos_per_prop_id(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        mean_pos_df = train_df[train_df['random_bool'] == 0][['prop_id', 'position']].groupby(['prop_id']).mean()
        mean_pos_df = add_suffix(mean_pos_df, suffix=f'_mean_on_prop_id', cols_to_exclude=['prop_id'])

        std_pos_df = train_df[train_df['random_bool'] == 0][['prop_id', 'position']].groupby(['prop_id']).std()
        std_pos_df = add_suffix(std_pos_df, suffix=f'_std_on_prop_id', cols_to_exclude=['prop_id'])

        test_df = pd.merge(test_df, std_pos_df, on='prop_id', how='left')
        test_df = pd.merge(test_df, mean_pos_df, on='prop_id', how='left')

        train_df.drop('position', axis=1, inplace=True)

        train_df = pd.merge(train_df, std_pos_df, on='prop_id', how='left')
        train_df = pd.merge(train_df, mean_pos_df, on='prop_id', how='left')

        return train_df, test_df
        # # Mean pos = 16.856
        # mean_pos_df = _calc_average_pos_per_prop_id(train_df)
        # #train_df = train_df.merge(mean_pos_df, how='left')
        # test_df = test_df.reset_index().merge(mean_pos_df, how='left').set_index('index')
        #
        #
        # return train_df, test_df

    @staticmethod
    def click_and_booking_prob(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for feature in ['booking', 'click']:
            data_sum = train_df[['prop_id', f'{feature}_bool']].groupby('prop_id')[f'{feature}_bool'].sum().reset_index(name='sum')
            train_df = pd.merge(train_df, data_sum, on='prop_id', how='left')

            data_count = train_df[['prop_id', f'{feature}_bool']].groupby('prop_id')[f'{feature}_bool'].count().reset_index(name='count')
            train_df = pd.merge(train_df, data_count, on='prop_id', how='left')

            train_df[f'{feature}_prob'] = (train_df['sum'] - train_df[f'{feature}_bool'])/ (train_df['count'] - 1)
            train_df.drop(['sum', 'count'], axis=1, inplace=True)

            test_df = pd.merge(test_df, train_df[[f'{feature}_prob', 'prop_id']].drop_duplicates(subset=['prop_id']), on=['prop_id'], how='left')
        return train_df, test_df


    @staticmethod
    def drop_cols(df: pd.DataFrame, exclude_cols_to_drop: List[str] = []) -> pd.DataFrame:
        cols_to_drop_copy = COLS_TO_DROP.copy()
        for col in exclude_cols_to_drop:
            cols_to_drop_copy.remove(col)
        df.drop(cols_to_drop_copy, axis=1, errors='ignore', inplace=True)
        return df

    @staticmethod
    def extract_booking_month(df: pd.DataFrame) -> pd.DataFrame:
        df['date_time'] = pd.to_datetime(df['date_time'], format='%Y-%m-%d')
        df['booking_month'] = (df['date_time'] + pd.to_timedelta(df['srch_booking_window'], unit='D')).apply(
            lambda date: date.month)
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder_df = pd.DataFrame(encoder.fit_transform(df[['booking_month']]).toarray())
        df = df.join(encoder_df)
        df.drop(['booking_month'], axis=1, inplace=True)
        return df

    @staticmethod
    def impute_competitors(df: pd.DataFrame) -> pd.DataFrame:
        cols = df.filter(regex='comp').columns
        df[cols] = df[cols].fillna(0)
        return df

    @staticmethod
    def impute(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_cols = test_df.columns
        train_copy = train_df.copy()[test_cols]
        combined_df = pd.concat([train_copy, test_df])
        imputations = {
            'prop_location_score2': 0,
            'srch_query_affinity_score': 'median',
            'visitor_hist_starrating': 'mean',
            'visitor_hist_adr_usd': 'mean',
            'orig_destination_distance': 'median',
            'comp2_rate': 0,
            'comp2_rate_percent_diff': 0,
            'comp5_rate': 0,
            'comp7_rate_percent_diff': 0,
            'comp8_rate': 0,
            'prop_review_score': 'mean',
            'position_mean': 16
        }
        for col, val in imputations.items():
            if val == 'median':
                train_df[col] = train_df[col].fillna(combined_df[col].median())
                test_df[col] = test_df[col].fillna(combined_df[col].median())
            elif val == 'mean':
                train_df[col] = train_df[col].fillna(combined_df[col].mean())
                test_df[col] = test_df[col].fillna(combined_df[col].mean())
            else:
                train_df[col] = train_df[col].fillna(val)
                test_df[col] = test_df[col].fillna(val)

        return train_df, test_df



def _calc_average_pos_per_prop_id(train_df: pd.DataFrame) -> None:
    pos_df = train_df[['prop_id', 'position']]
    # mean_pos = pos_df['position'].mean()
    df_mean = pos_df.groupby('prop_id', as_index=False).mean()
    #df_mean.rename(columns={'position': 'position_mean'}, inplace=True)
    return df_mean


if __name__ == "__main__":
    train_df, test_df = DataLoader.load_train_test_dfs()
    train_pos_test = train_df[['position']]
    train_pos_test = PreprocessorTwo.exp_pos(train_pos_test)
    print(train_pos_test)
