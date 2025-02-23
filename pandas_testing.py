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

print(df.head()) # Get First 5 entries(num can be changed - n)
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
testing = [
    ("df[0:3]",df[0:3]),
    ('df["20130102":"20130104"]',df["20130102":"20130104"]),
    ("df.loc[dates[0]]",df.loc[dates[0]]),
    ('df.loc[:, ["A", "B"]]', df.loc[:, ["A", "B"]]),
    ('df.loc["20130102":"20130104", ["A", "B"]]',df.loc["20130102":"20130104", ["A", "B"]]),
    ('df.loc[dates[0], "A"]',df.loc[dates[0], "A"]),
    ('df.at[dates[0], "A"]',df.at[dates[0], "A"]),
    ()
]
for text, value in testing:
    print(text, "\n", value)
