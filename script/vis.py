import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
songs = pd.read_csv('songs.csv')
members = pd.read_csv('members.csv')

# print(train.head())
# print(test.head())
# print(songs.head())
# print(members.head())

# print(train.info())
# print(test.info())
# print(songs.info())
# print(members.info())

# fig, ax = plt.subplots(1, 2, figsize=(11, 6))
# sns.countplot(y='source_type', data=train, ax=ax[0])
# sns.countplot(y='source_system_tab', data=train, ax=ax[1])
# plt.tight_layout()
# fig.subplots_adjust(wspace=0.5)
# fig.show()


# fig, ax = plt.subplots(2, 2, figsize=(11, 6))

# sns.countplot(y='source_type', data=train, ax=ax[0, 0])
# sns.countplot(y='source_system_tab', data=train, ax=ax[0, 1])
# sns.countplot(y='source_screen_name', data=train, ax=ax[1, 0])

# pp= pd.value_counts(members.gender)
# pp.plot.pie(startangle=90, autopct='%1.1f%%', shadow= False, explode= (0.05,0.05))
# ax[1,1].axis('off')

# plt.tight_layout()
# fig.subplots_adjust(wspace=0.5)
# fig.show()

# merging train datasets
train_members = pd.merge(train, members, on='msno', how='inner')
train_merged = pd.merge(train_members, songs, on='song_id', how='outer')
# print(train_merged.head())
# print(len(train_merged.columns))

# merging test datasets
test_members = pd.merge(test, members, on='msno', how='inner')
test_merged = pd.merge(test_members, songs, on='song_id', how='outer')
# print(test_merged.head())
# print(len(test_merged.columns))


# def check_missing_values(df):
#     if (df.isnull().values.any() == True):
#         columns_with_Nan = df.columns[df.isnull().any()].tolist()
#         for col in columns_with_Nan:
#             print('%s : %d' % (col, df[col].isnull().sum()))
# check_missing_values(train_merged)


# Replace missing float values with -5
def replace_Nan_non_object(df):
    object_cols = list(df.select_dtypes(include=['float']).columns)
    for col in object_cols:
        df[col] = df[col].fillna(-5)


replace_Nan_non_object(train_merged)
replace_Nan_non_object(test_merged)


# Replace missing object values with ' '
def replace_Nan_object(df):
    object_cols = list(df.select_dtypes(include=['object']).columns)
    for col in object_cols:
        df[col] = df[col].fillna(' ')


replace_Nan_object(train_merged)
replace_Nan_object(test_merged)

train_merged = train_merged[train_merged.target != -5]


# msno_target = train_merged.groupby('target').aggregate(
#     {'msno': 'count'}).reset_index()
# msno_city = train_merged.groupby('city').aggregate(
#     {'msno': 'count'}).reset_index()

# fig, ax = plt.subplots(2, 1, figsize=(11, 6))
# sns.barplot(x='target', y='msno', data=msno_target, ax=ax[0])
# sns.barplot(x='city', y='msno', data=msno_city, ax=ax[1])
# plt.subplots_adjust(hspace=0.3, wspace=0.3, rowspan=2)
# plt.tight_layout()
# fig.show()
