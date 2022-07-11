import pandas as pd

ORIG_TRAIN_PATH = '../../data/original/train.csv'
GIVEN_TEST_PATH = '../../data/test_set_VU_DM.csv'

orig_train_df = pd.read_csv(ORIG_TRAIN_PATH)
given_test_df = pd.read_csv(GIVEN_TEST_PATH)
print("Loading finished")
dropped_orig = orig_train_df.drop_duplicates(subset=['prop_id', 'date_time', 'site_id', 'visitor_location_country_id'])
merged = pd.merge(dropped_orig, given_test_df, on=['prop_id', 'date_time', 'site_id', 'visitor_location_country_id'])
print("First merge done")
merged_new = merged[['srch_id_y', 'date_time', 'booking_bool', 'click_bool', 'prop_id']]
merged_new = merged_new.rename(columns={"srch_id_y": "srch_id"})

merged_new = merged_new.rename(columns={"srch_id_y": "srch_id"})

print("Before new merge")
given_test_with_bools = pd.merge(given_test_df, merged_new, on=['srch_id', 'date_time', 'prop_id'])
print("After new merge")
given_test_with_bools.to_csv('testset_reconstructed.csv', index=False)
print("Done")