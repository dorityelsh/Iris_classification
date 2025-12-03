from sklearn import datasets
import pandas as pd
datasets.load_iris()
import matplotlib.pyplot as plt
import seaborn as sns


#load the dataset into a pandas DataFrame from sklearn built-in datasets
data = datasets.load_iris()
print(data.keys())
print(data["DESCR"])
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

#Explore the dataset]

print (df.describe())
print(df.head())

### Distribution of features and target variable
df["target_name"] = df["target"].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
sns.pairplot(df, hue='target_name', markers=["o", "s", "D"],vars=data.feature_names)
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()  
