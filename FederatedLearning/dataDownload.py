from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(iris.data)
# shuffle the dataframe
df["label"] = iris.target
df = df.sample(frac=1)
print(df)
df[:30].to_csv("dataset/iris1.csv", index=False)  # Client 1 data
df[30:60].to_csv("dataset/iris2.csv", index=False)  # Client 2 data
df[60:90].to_csv("dataset/iris3.csv", index=False)  # Client 3 data
df[90:].to_csv("dataset/iris4.csv", index=False)  # Central Test Data
