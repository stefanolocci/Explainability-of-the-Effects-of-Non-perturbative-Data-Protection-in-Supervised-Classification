import pandas as pd
import anonypy


def fix_sep(lst):
    if isinstance(lst, list):
        return lst[0]
    else:
        return lst


# Load the Poker Hand dataset from CSV file
ph_dataset_path = "data/poker_hand/poker_hand_train.csv"
cah_dataset_path = "data/cahousing/cahousing_id.csv"
diab_dataset_path = "data/diabetes/diabetes_id.csv"
df = pd.read_csv(diab_dataset_path, sep=',')
# ph_qi_attributes = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5"]
# cah_qi_attributes = ["longitude", "latitude", "housing_median_age", "median_income", "median_house_value"]
diab_qi_attributes = ['age', 'hypertension', 'HbA1c_level', 'blood_glucose_level']
# ph_sensitive_attribute = "Label"
# cah_sensitive_attribute = "ocean_proximity"
diab_sensitive_attribute = "diabetes"
k_values = [2, 5, 10, 20, 50, 70, 100]
l_values = [2, 3]
for k in k_values:
    # for l in l_values:
    p = anonypy.Preserver(df, diab_qi_attributes, diab_sensitive_attribute)
    # anon_rows = p.anonymize_l_diversity(k, l)
    anon_rows = p.anonymize_k_anonymity(k)

    anon_df = pd.DataFrame(anon_rows)
    anon_df = anon_df.applymap(lambda x: fix_sep(x))

    anon_df = anon_df.loc[anon_df.index.repeat(anon_df['count'])].reset_index(drop=True)
    anon_df = anon_df.drop('count', axis=1)

    # print(df)

    # print(anon_df.to_csv('anon_poker_hand.csv'))
    # Output the anonymized dataset to CSV file
    # anon_df.to_csv(f"anonymized_data/poker_hand/mondrian_ldiv/anonypy/poker_hand_anon_k_{k}_l_{l}", sep=';', index=False)
    # anon_df.to_csv(f"anonymized_data/diabetes/mondrian_ldiv/anonypy/diabetes_anon_k_{k}_l_{l}", sep=';', index=False)
    anon_df.to_csv(f"results/diabetes/mondrian/anonypy/diabetes_anon_k_{k}", sep=';', index=False)
