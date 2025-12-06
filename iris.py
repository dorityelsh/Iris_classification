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
df["target_name"] = df["target"].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
#Explore the dataset]
def explore_data(df):
  
    print (df.describe())
    print(df.head())

    ### Distribution of features and target variable
    
    sns.pairplot(df, hue='target_name', markers=["o", "s", "D"],vars=data.feature_names)
    plt.suptitle('Pairplot of Iris Dataset', y=1.02)
    plt.show()  
    return 

#explore_data(df)


#train test split
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.25)
print(f"Train set size: {df_train.shape}, Test set size: {df_test.shape}")

#PREPARING DATA FOR MODELING
x_train = df_train.drop(columns=['target','target_name']).values     
y_train = df_train['target']

#manually model 
def single_feature_modeling(petal_length):
    """predict the Iris species using a single feature - petal length"""
    if petal_length < 2.5:
        return 0  # setosa
    elif petal_length < 4.8:
        return 1  # versicolor
    else:
        return 2  # virginica
manual_predictions = [single_feature_modeling(pl) for pl in x_train[:,2]]  # petal length is the 3rd feature