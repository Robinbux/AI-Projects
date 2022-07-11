from typing import List
import pandas as pd
import os
import datetime
import numpy.typing as npt
from pathlib import Path

from src_clean.dataloader import DataLoader
from src_clean.model.model import Model

RANKINGS_DIR = f'{Path(__file__).resolve().parent.parent.parent}/rankings'

class Ranker:

    def __init__(self):
        _, self.test_df = DataLoader.load_train_test_dfs()
        self.test_df_features = self.test_df.copy()
        self.search_ids = self.test_df_features.pop('srch_id')
        self.prop_ids = self.test_df_features.pop('prop_id')

    def make_ranking_from_model(self, model: Model, model_name: str, cols: List[str] = None) -> None:
        if cols:
            self.make_ranking_from_prediction(model.predict(self.test_df_features[cols].values), model_name)
        else:
            self.make_ranking_from_prediction(model.predict(self.test_df_features.values), model_name)

    def make_ranking_from_prediction(self, prediction: npt.NDArray, model_name: str) -> str:
        new_rank_test_df = self.test_df.copy()
        new_rank_test_df['rel_pred'] = prediction
        new_rank_test_df.sort_values(by=['srch_id', 'rel_pred'], ascending=[True, False], inplace=True)

        result_df = pd.DataFrame(columns=['srch_id', 'prop_id'])
        result_df['srch_id'] = new_rank_test_df['srch_id']
        result_df['prop_id'] = new_rank_test_df['prop_id']

        if not os.path.exists(f'{RANKINGS_DIR}/{model_name}'):
            os.makedirs(f'{RANKINGS_DIR}/{model_name}')

        now = datetime.datetime.now().strftime("%d_%m_%Y-%M_%H")
        file_path = f'{RANKINGS_DIR}/{model_name}/ranking_{now}.csv'
        result_df.to_csv(file_path, index=False)
        return file_path

