# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1: Read the given Data.
STEP 2: Clean the Data Set using Data Cleaning Process.
STEP 3: Apply Feature Scaling for the feature in the data set.
STEP 4: Apply Feature Selection for the feature in the data set.
STEP 5: Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1. Filter Method
2. Wrapper Method
3. Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data

data.isnull().sum()

missing=data[data.isnull().any(axis=1)]
missing

data2=data.dropna(axis=0)
data2

sal=data['SalStat']
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,
              'greater than 50,000':1})
print(data2['SalStat'])

sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs

data2

new_data=pd.get_dummies(data2, drop_first=True)
new_data

columns_list=list(new_data.columns)
print(columns_list)

new_data=pd.get_dummies(data2, drop_first=True)
new_data

columns_list=list(new_data.columns)
print(columns_list)

features=list(set(columns_list)-set(['SalStat']))
print(features)

y=new_data['SalStat'].values
print(y)

x=new_data[features].values
print(x)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

print("Misclassified Samples : %d" % (test_y !=prediction).sum())

data.shape

data.shape

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)

chi2,p, _, _=chi2_contingency(contigency_table)
print(f"chi-Square Statistic:{chi2}")
print(f"P-value:{p}")

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/273ed690-3c9d-4d13-b90c-889d1045743b)

![image](https://github.com/user-attachments/assets/b29c513b-8e1a-4edf-991e-4e5180a04682)

![image](https://github.com/user-attachments/assets/49ac6619-4d50-495b-8157-5f094ca9f16f)

![image](https://github.com/user-attachments/assets/9ca80531-3bdb-409b-8222-d3e9fa40162d)

![image](https://github.com/user-attachments/assets/90da8d0c-17ee-4972-a3cf-1de6b9584507)

![image](https://github.com/user-attachments/assets/cf06a3f2-00a9-4b62-ac32-622ba69f2e45)

![image](https://github.com/user-attachments/assets/cf541996-6c63-4cee-8de1-ad07b2ef1ead)

![image](https://github.com/user-attachments/assets/bd47d21d-1f13-4ed4-bf22-21e65fccc2c1)

![image](https://github.com/user-attachments/assets/0c3eb551-8668-4ae1-9d51-956afc712885)

![image](https://github.com/user-attachments/assets/a6db6a8d-cebe-428d-9f8e-0250061047c2)

![image](https://github.com/user-attachments/assets/778f0a48-5514-48f9-9d62-9ff39bcfdeac)

![image](https://github.com/user-attachments/assets/1f11a269-3e36-4b43-b693-d56e9600e886)

![image](https://github.com/user-attachments/assets/c29e9377-85ba-4de0-8270-25048e95a95f)

![image](https://github.com/user-attachments/assets/3f1604b4-0f74-46f0-84a1-72b471c498be)

![image](https://github.com/user-attachments/assets/dd219b3d-d04f-4453-9f57-b05631d08d32)

![image](https://github.com/user-attachments/assets/8dbbc0d3-24d1-4d19-aa73-00559bf4f61f)

![image](https://github.com/user-attachments/assets/a7448405-74ea-4107-a971-8f0496f1eaa8)

![image](https://github.com/user-attachments/assets/b0bec49e-b96c-4194-9494-98bc105d9555)

![image](https://github.com/user-attachments/assets/49fb8ab2-2d29-4408-9e37-0a49c3d00513)

![image](https://github.com/user-attachments/assets/512b59db-f313-46dd-8ac6-1cde448d5546)

# RESULT:
Thus the code to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is implemented successfully.
