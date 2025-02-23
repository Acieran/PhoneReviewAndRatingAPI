import numpy as np

import pandas as pd

s = pd.Series([1, 3, 5, np.nan, 6, 8])

dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)

#                    A         B         C         D
# 2013-01-01 -0.064563  1.276014  0.235959 -0.970489
# 2013-01-02  0.453171  1.477960  0.027034  0.917174
# 2013-01-03  0.338768 -1.426953 -1.090422  0.998539
# 2013-01-04 -0.764782  0.794198  0.306514 -1.426409
# 2013-01-05 -0.366245 -0.638924  1.351295  0.029362
# 2013-01-06 -0.640060  0.720118 -0.166800  0.702348

print(df.head(6)) # Get First 5 entries(num can be changed - n)
print(df.tail()) # Get Last 5 entries(num can be changed - n)
print(df.index) # Get All available
print(df.columns) # Get All Column names
nump_df = df.to_numpy() #Convert to array or array of arrays
print(nump_df)
print(df.describe()) # Basic Description
print(df.T) # Transpose
print(df.sort_index(axis=1, ascending=False)) # Sort by index(axis - 0) or column(axis - 1)
print(df.sort_values(by="2013-01-01", axis=1)) # Sort Values by specific column
print(df["B"])
df2 = df.copy()

df2["E"] = ["one", "one", "two", "three", "four", "three"]

selection = [
    ("df[0:3]",df[0:3]), # Row selection(Index)
    ('df["20130102":"20130104"]',df["20130102":"20130104"]), # Row selection(Values)
    ("df.loc[dates[0]]",df.loc[dates[0]]), # Same as above
    ('df.loc[:, ["A", "B"]]', df.loc[:, ["A", "B"]]), # Selection, : - get all rows, than select specific columns
    ('df.loc["20130102":"20130104", ["A", "B"]]',df.loc["20130102":"20130104", ["A", "B"]]), # Same as above, but specified rows
    ('df.loc[dates[0], "A"]',df.loc[dates[0], "A"]), # Get 1 item
    ('df.at[dates[0], "A"]',df.at[dates[0], "A"]), # Get 1 item, Faster, user this
    ('df.iloc[3]', df.iloc[3]), # Select by Position(row)
    ('df.iloc[3:5, 0:2]',df.iloc[3:5, 0:2]), # Select by Position Row than Column
    ('df.iloc[[1, 2, 4], [0, 2]]', df.iloc[[1, 2, 4], [0, 2]]), # Same, just specified indexes
    ('df.iloc[1:3, :]',df.iloc[1:3, :]), # Same, but all Columns
    ('df.iloc[:, 1:3]',df.iloc[:, 1:3]), # Same, but all rows
    ('df.iloc[1, 1]', df.iloc[1, 1]), # Get 1 item
    ('df.iat[1, 1]', df.iat[1, 1]), # Get 1 item, Faster, use this
    ('df[df["A"] > 0]', df[df["A"] > 0]), # Select rows where df.A is greater than 0.
    ('df[df > 0]',df[df > 0]), # Selecting values from a DataFrame where a boolean condition is met:
    ('df2',df2),
    ('df2[df2["E"].isin(["two", "four"])]', df2[df2["E"].isin(["two", "four", "three"])]), # issin is checking whenever values are on of
]
for text, value in selection:
    print(text, "\n", value)

# Setting values
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
df["F"] = s1
df.at[dates[0], "A"] = 0
print('df.at[dates[0], "A"]',"\n",df)
df.iat[0, 1] = 0
print('df.iat[0, 1] = 0', "\n", df)
df.loc[:, "D"] = np.array([5] * len(df))
print('df.loc[:, "D"] = np.array([5] * len(df))', "\n",df)

df2 = df.copy()
df2[df2 > 0] = -df2
print('df2[df2 > 0] = -df2', "\n", df2)
# Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of the data:
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
df1.loc[dates[0] : dates[1], "E"] = 1
print('df1.loc[dates[0] : dates[1], "E"] = 1',df1, sep="\n")
print('df1.dropna(how="any")',"\n",df1.dropna(how="any")) # DataFrame.dropna() drops any rows that have missing data
print('df1.fillna(value=5)',"\n",df1.fillna(value=5)) # DataFrame.fillna() fills missing data
print('pd.isna(df1)', "\n",pd.isna(df1)) # isna() gets the boolean mask where values are nan:

# Operations
print('df.mean()', "\n", df.mean()) # Calculate the mean value for each column:
print('df.mean(axis=1)',"\n",df.mean(axis=1)) #Calculate the mean value for each row:
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print('pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)', "\n",s)
print('df.sub(s, axis="index")', "\n", df.sub(s, axis="index"))
print('df.agg(lambda x: np.mean(x) * 5.6)', "\n",df.agg(lambda x: np.mean(x) * 5.6)) # DataFrame.agg() applies a user defined function that reduces its result respectively.
print('df.transform(lambda x: x * 101.2)', "\n", df.transform(lambda x: x * 101.2)) # DataFrame.transform() applies a user defined function that broadcasts its result respectively.

s = pd.Series(np.random.randint(0, 7, size=10))
print('pd.Series(np.random.randint(0, 7, size=10))', "\n", s)
print('s.value_counts()', "\n",s.value_counts()) # Counts number of occurrences

s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
print('s.str.lower()', "\n", s.str.lower()) # U can use string methods
