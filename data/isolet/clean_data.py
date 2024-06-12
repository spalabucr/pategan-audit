import string
import pandas as pd


# read csv
dtype = {f"f{i}": float for i in range(1, 618)} | {"class": int}

df_path = "data.csv.gz"
test_df_path = "test_data.csv.gz"
train_df = pd.read_csv(df_path, dtype=dtype)
test_df = pd.read_csv(df_path, dtype=dtype)

# binarize label
vowels = [i + 1 for i, ch in enumerate(string.ascii_lowercase) if ch in "aeiou"]

vowels_idx = train_df["class"].isin(vowels)
train_df.loc[vowels_idx, "class"], train_df.loc[~vowels_idx, "class"] = 1, 0

vowels_idx = test_df["class"].isin(vowels)
test_df.loc[vowels_idx, "class"], test_df.loc[~vowels_idx, "class"] = 1, 0

# save adult
train_df.to_pickle("train_isolet.pkl.gz", compression="gzip")
test_df.to_pickle("test_isolet.pkl.gz", compression="gzip")
