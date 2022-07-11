from src_clean.dataloader.dataloader import DataLoader
from src_clean.preprocessor.prepro_new import PreprocessorTwo

print("START")
train_df, test_df = DataLoader.load_train_test_dfs()
train_df = PreprocessorTwo.remove_outliers(train_df)
train_df = PreprocessorTwo.relevance(train_df)

train_df, test_df = PreprocessorTwo.calculate_statistics(train_df, test_df)
train_df, test_df = PreprocessorTwo.mean_pos_per_prop_id(train_df, test_df)
#train_df, test_df = PreprocessorTwo.click_and_booking_prob(train_df, test_df)

train_df = PreprocessorTwo.extract_booking_month(train_df)
test_df = PreprocessorTwo.extract_booking_month(test_df)
print("Before norm")
train_df = PreprocessorTwo.feature_normalization(train_df)
test_df = PreprocessorTwo.feature_normalization(test_df)
print("After norm")
print(test_df.columns)
# train_df = PreprocessorTwo.impute_competitors(train_df) nope
# test_df = PreprocessorTwo.impute_competitors(test_df) nope

train_df = PreprocessorTwo.drop_cols(train_df, exclude_cols_to_drop=['srch_id', 'prop_id'])
test_df = PreprocessorTwo.drop_cols(test_df)

from src_clean.ranker.ranker import Ranker
from src_clean.score.ndcg import calculate_ndcg
import lightgbm
import numpy as np
X_train, X_valid, y_train, y_valid = DataLoader.split_df_into_train_and_val_batches(train_df, validation_size=0.1)
group_train = [group.shape[0] for group in X_train]
group_val = [group.shape[0] for group in X_valid]

params = {'learning_rate': 0.08,
          'num_iterations': 500,
          }

model = lightgbm.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    **params
)
model.fit(
    X=np.vstack(X_train),
    y=np.hstack(y_train),
    group=group_train,
    eval_set=[(np.vstack(X_valid), np.hstack(y_valid))],
    eval_group=[group_val],
    eval_at=5,
    verbose=10
)
pred = model.predict(test_df.to_numpy())
ranker = Ranker()
ranking_file_path = ranker.make_ranking_from_prediction(
    pred, model_name="LightGBM"
)
ndcg_score = calculate_ndcg(ranking_file_path)
print(f"Final NDCG: {ndcg_score}")
print(f'File path: {ranking_file_path}')