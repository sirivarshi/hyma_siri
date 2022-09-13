import numpy
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv(r"C:\Users\HY20367012\Downloads\machine_learning_model_using_flask_web_framework-master\adult.csv")
df.to_csv("adults.csv")
df = pd.read_csv("adults.csv")
print(df.head())
print(df.columns)
df = df.drop(["fnlwgt","educational-num"],axis = 1)
print(df.columns)
for c in  df.columns:
    df = df.replace("?",numpy.NAN)
print(df.head())
df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
print(df.head())
(df.replace(['Divorced', 'Married-AF-spouse',
            'Married-civ-spouse', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Widowed'],
           ['divorced', 'married', 'married', 'married',
            'not married', 'not married', 'not married'], inplace=True))
categorical_columns = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                'relationship', 'gender', 'native-country', 'income']
labelEncoder = preprocessing.LabelEncoder()
mapping_dict = {}
for col in categorical_columns:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_,labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col] = le_name_mapping
print(mapping_dict)
x = df.values[:,0:12]
y = df.values[:,12]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=100)
dt_clf_gini = DecisionTreeClassifier(criterion = "gini",
                                     random_state = 100,
                                     max_depth = 5,
                                     min_samples_leaf = 5)
dt_clf_gini.fit(X_train, y_train)
y_pred_gini = dt_clf_gini.predict(X_test)
print ("Decision Tree using Gini Index\nAccuracy is ",
             accuracy_score(y_test, y_pred_gini)*100 )


with open('model_pkl', 'wb') as files:
    pickle.dump(dt_clf_gini, files)










