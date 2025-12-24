from sklearn import datasets
import pandas as pd
datasets.load_iris()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,cross_val_predict
import numpy as np
from sklearn.ensembles import RandomForestClassifier


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
manual_predictions == y_train
manual_model_accuracy = np.mean(manual_predictions == y_train)
print(f"Manual model accuracy: {manual_model_accuracy*100:.2f}%")

#modling logistic regression
model =LogisticRegression()

# split training data into training and validation sets to evaluate model performance not on training data
#xt stands for x train, xv for x validation
xt, xv, yt, yv = train_test_split(x_train,y_train,test_size=0.25)

model.fit(xt, yt) # Train the model 
y_pred = model.predict(xv) # Use the trained model to predict labels for new data
np.mean(y_pred == yv) # same as model.score(xv,yv) Calculate the accuracy by comparing predicted labels to true labels

# model.score(x_train, y_train) - you never want to evalute the model on the same data you trained it on
print(f"Logistic Regression model accuracy on validation set: {np.mean(y_pred == yv)*100:.2f}%")

#using cross validation to evaluate the model
"""cross_val_predict generates out-of-sample predictions for each training sample 
by training the model on k-1 folds and predicting the held-out fold"""

model=LogisticRegression(max_iter=200)
accouracies = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
print(np.mean(accouracies))
y_pred = cross_val_predict(model, x_train, y_train, cv=5)
predicted_correctly_mask = y_pred == y_train
x_train[predicted_correctly_mask] # correctly predicted samples
print(x_train[~predicted_correctly_mask]) # incorrectly predicted samples

# prepare data for plot
df_predictions = df_train.copy()
df_predictions["Correct_Prediction"] = predicted_correctly_mask
df_predictions['prediction'] = y_pred
df_predictions['prediction_label'] = df_predictions['prediction'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
sns.scatterplot(data=df_predictions, x='petal length (cm)', y='petal width (cm)', hue='prediction_label', style='Correct_Prediction', palette='deep')
sns.scatterplot(data=df_predictions, x='petal length (cm)', y='petal width (cm)', hue='target_name', style='Correct_Prediction', palette='deep')

def plot_incorrect_predictions(df_predictions,x_axis_feature, y_axis_feature):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    sns.scatterplot(data=df_predictions, x=x_axis_feature, y=y_axis_feature, hue='prediction_label', ax =axs[0])
    sns.scatterplot(data=df_predictions, x=x_axis_feature, y=y_axis_feature, hue='target_name', ax =axs[1]) 
    sns.scatterplot(data=df_predictions, x=x_axis_feature, y=y_axis_feature, hue='correct_prediction', ax =axs[2])
    axs[3].set_visible(False)
    plt.show()

plot_incorrect_predictions(df_predictions,'petal length (cm)', 'petal width (cm)')

# Model Tuning with Random Forest
# Trying to determine the parameters of your model that maximize performance






# Try another model - Random Forest
# model = RandomForestClassifier()
# accs=cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
# print(f"Random Forest model accuracy on validation set: {np.mean(accs)*100:.2f}%") 


  