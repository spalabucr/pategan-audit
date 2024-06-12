import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# read csv
dtypes = {
    "Age": int,
    "N_sex_partners": int,
    "First_sex_intercourse": int,
    "N_pregnancies": int,
    "Smokes": int,
    "Smokes_years": float,
    "Smokes_packs_year": float,
    "Hormonal_Contraceptives": int,
    "Hormonal_Contraceptives_years": float,
    "IUD": int,
    "IUD_years": float,
    "STDs": int,
    "STDs_number": int,
    "STDs_condylomatosis": int,
    # "STDs_cervical_condylomatosis": int,
    "STDs_vaginal_condylomatosis": int,
    "STDs_vulvo_perineal_condylomatosis": int,
    "STDs_syphilis": int,
    # "STDs_pelvic_inflammatory_disease": int,
    # "STDs_genital_herpes": int,
    # "STDs_molluscum_contagiosum": int,
    # "STDs_AIDS": int,
    "STDs_HIV": int,
    # "STDs_Hepatitis_B": int,
    # "STDs_HPV": int,
    "STDs_N_diagnosis": int,
    "STDs_Time_since_first_diagnosis": int,
    "STDs_Time_since_last_diagnosis": int,
    "Dx_Cancer": int,
    "Dx_CIN": int,
    "Dx_HPV": int,
    "Dx": int,
    "Hinselmann": int,
    "Schiller": int,
    "Citology": int,
    "Biopsy": int,
}

df_path = "data.csv.gz"
df = pd.read_csv(df_path)

# drop columns
df = df.replace('?', np.nan)
df = df.drop(["STDs_cervical_condylomatosis",
              "STDs_pelvic_inflammatory_disease",
              "STDs_genital_herpes",
              "STDs_molluscum_contagiosum",
              "STDs_AIDS",
              "STDs_Hepatitis_B",
              "STDs_HPV",
              ], axis=1)

# fill na
df = df.apply(pd.to_numeric)
df = df.fillna(df.mode().iloc[0])

# apply dttypes
df = df.astype(dtypes)

# train test split
train_df, test_df = train_test_split(df, stratify=df["Biopsy"], test_size=0.2, random_state=13)

# save adult
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_pickle("train_cervical_cancer.pkl.gz", compression="gzip")
test_df.to_pickle("test_cervical_cancer.pkl.gz", compression="gzip")
