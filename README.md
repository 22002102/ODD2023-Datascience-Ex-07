# Ex-07 Feature Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```
 NAME : SANJAY S
 REG NO : 212222230132
```
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("titanic_dataset.csv")
data.head()
data.isnull().sum()
sns.heatmap(data.isnull(),cbar=False)
plt.title("sns.heatmap(data.isnull(),cbar=False)")
plt.show()
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())
data['Embarked']=data['Embarked'].fillna('S')
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)
sns.heatmap(data.corr(),annot=True,fmt= '.1f',ax=ax)
plt.title("HeatMap")
plt.show()
sns.heatmap(data.isnull(),cbar=False)            
plt.show()
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)                  
plt.show()
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()
plt.scatter(data.Survived,data.Age,alpha=0.1)
plt.title("Age with Survived")                                
plt.show()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = data.drop("Survived",axis=1)
y = data["Survived"]
mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix])
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values
#Build test and training test
X_train,X_test,y_train,y_test = train_test_split
      (features,target,test_size=0.3,random_state=42)
my_forest=RandomForestClassifier(max_depth=5,min_samples_split=10,
                n_estimators=500,random_state=5,criterion='entropy')
my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)
print("Random forest score: ",accuracy_score(y_test,target_predict))
from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
```
# OUPUT:
![image](https://github.com/22002102/ODD2023-Datascience-Ex-07/assets/119091638/c7889161-aff2-4c45-99ab-686fc239f017)
<br>


![image](https://github.com/22002102/ODD2023-Datascience-Ex-07/assets/119091638/36ccd788-135d-47d3-b10e-59bfe121fa47)
![image](https://github.com/22002102/ODD2023-Datascience-Ex-07/assets/119091638/cd9ee3a7-a0f0-410a-b526-96b089719a36)
<br>


![image](https://github.com/22002102/ODD2023-Datascience-Ex-07/assets/119091638/6eec79f3-7be6-4cce-8290-94183776a317)
<br>

![image](https://github.com/22002102/ODD2023-Datascience-Ex-07/assets/119091638/a7f05495-ffbd-4e0c-8725-a900be4bbca0)
<br>

![image](https://github.com/22002102/ODD2023-Datascience-Ex-07/assets/119091638/f6738c38-8d31-4ea4-81cc-12e20ee35f0a)
<br>

![image](https://github.com/22002102/ODD2023-Datascience-Ex-07/assets/119091638/e3ce8a3b-b483-4bdb-bbdc-86e824334469)
<br>

![image](https://github.com/22002102/ODD2023-Datascience-Ex-07/assets/119091638/e503c9f6-44f4-4865-8b5e-aaf80a0303fb)
<br>

![image](https://github.com/22002102/ODD2023-Datascience-Ex-07/assets/119091638/75d8e5b4-6338-4b8a-b886-24e82bca08e0)
<br>

# RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
