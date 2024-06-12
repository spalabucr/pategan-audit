import pandas as pd

from sklearn.model_selection import train_test_split


# read csv
dtypes = {
    "V1": float,
    "V2": float,
    "V3": float,
    "V4": float,
    "V5": float,
    "V6": float,
    "V7": float,
    "V8": float,
    "V9": float,
    "V10": float,
    "V11": float,
    "V12": float,
    "V13": float,
    "V14": float,
    "V15": float,
    "V16": float,
    "V17": float,
    "V18": float,
    "V19": float,
    "V20": float,
    "V21": float,
    "V22": float,
    "V23": float,
    "V24": float,
    "V25": float,
    "V26": float,
    "V27": float,
    "V28": float,
    "Amount": float,
    "Class": int
}

df_path = "data.csv.gz"
df = pd.read_csv(df_path, dtype=dtypes)

# drop Time
df = df.drop(['Time'], axis=1)

# reduce space
df = df.astype("float16")
df["Class"] = df["Class"].astype("int16")

# train test split
train_df, test_df = train_test_split(df, stratify=df["Class"], test_size=0.2, random_state=13)

# save adult
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_pickle("train_credit.pkl.gz", compression="gzip")
test_df.to_pickle("test_credit.pkl.gz", compression="gzip")
