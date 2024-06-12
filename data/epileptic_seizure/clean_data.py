import pandas as pd

from sklearn.model_selection import train_test_split


# read csv
dtype = "int"

df_path = "data.csv.gz"
df = pd.read_csv(df_path, dtype=dtype)

# binarize label
df.loc[df["y"].isin([2, 3, 4, 5]), "y"] = 0

# train test split
train_df, test_df = train_test_split(df, stratify=df["y"], test_size=0.2, random_state=13)

# save adult
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_pickle("train_epileptic_seizure.pkl.gz", compression="gzip")
test_df.to_pickle("test_epileptic_seizure.pkl.gz", compression="gzip")
