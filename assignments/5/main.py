import pandas as pd

df = pd.read_csv("new.data", sep=r"\s+", na_values=-9)

print(df.head(1))
print(df.dtypes)
print(df.head(2))