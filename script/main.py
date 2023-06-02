import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import warnings
from sklearn import model_selection, metrics, ensemble
from xgboost.sklearn import XGBClassifier

warnings.filterwarnings("ignore")

# load datasets
df = pd.read_csv("train.csv")
songs = pd.read_csv("songs.csv")
members = pd.read_csv("members.csv")

# merge datasets
df = pd.merge(df, songs, on="song_id", how="left")
df = pd.merge(df, members, on="msno", how="left")

del songs
del members

# df.info()
# df.isnull().sum()

# replace missing values
for i in df.select_dtypes(include=["object"]).columns:
    df[i][df[i].isnull()] = "unknown"
df = df.fillna(value=0)

# create dates
# registration time
df.registration_init_time = pd.to_datetime(
    df.registration_init_time, format="%Y%m%d", errors="ignore"
)
df["registration_init_time_year"] = df["registration_init_time"].dt.year
df["registration_init_time_month"] = df["registration_init_time"].dt.month
df["registration_init_time_day"] = df["registration_init_time"].dt.day

# expiration date
df.expiration_date = pd.to_datetime(
    df.expiration_date, format="%Y%m%d", errors="ignore"
)
df["expiration_date_year"] = df["expiration_date"].dt.year
df["expiration_date_month"] = df["expiration_date"].dt.month
df["expiration_date_day"] = df["expiration_date"].dt.day

# dates to category
df["registration_init_time"] = df["registration_init_time"].astype("category")
df["expiration_date"] = df["expiration_date"].astype("category")

# objects to category
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype("category")

# encoding all categorical features
for col in df.select_dtypes(include=["category"]).columns:
    df[col] = df[col].cat.codes

df = df.drop(["expiration_date", "lyricist"], axis=1)
# print(df.columns)

# train test split
target = df.pop("target")
train_data, test_data, train_labels, test_labels = model_selection.train_test_split(
    df, target, test_size=0.3
)
del df

# Random forest
model = ensemble.RandomForestClassifier(n_estimators=250, max_depth=25)
model.fit(train_data, train_labels)

model_pred = model.predict(test_data)
print(metrics.accuracy_score(test_labels, model_pred))

# XG boost
model1 = XGBClassifier(
    learning_rate=0.1, max_depth=15, min_child_weight=5, n_estimators=250
)
model1.fit(train_data, train_labels)

model1_pred = model1.predict(test_data)
print(metrics.accuracy_score(test_labels, model1_pred))
