#imports
#import
import numpy as np
import pandas as pd
import pickle
#import seaborn as sns
#import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score

#source
url = 'train.csv'

#reading the source
titanic = pd.read_csv(url)

#Replcaing null values
titanic.fillna(titanic.mean(), inplace=True)

# Doing EDA 
def age_convertor(Age):
    if ((Age<18) or (Age>70)):
        return 1
    else:
        return 2
    
    
def Parch_convertor(parch):
    if parch == 0:
        return 0
    elif parch ==1:
        return 1
    elif parch ==2:
        return 2
    else:
        return 3  

titanic['Sex_number'] = titanic.apply(lambda x :1 if (x['Sex'])=='male' else 0,axis=1)
titanic['Age_splitter'] = titanic.apply(lambda x:age_convertor(x['Age']),axis=1)
titanic['Alone'] = titanic.apply(lambda x :1 if (x['SibSp']>0) else 0,axis=1)
titanic['Parch_splitter'] = titanic.apply(lambda x:Parch_convertor(x['Parch']),axis=1)

#building model with selected columns along with target label
train1 = titanic[['Pclass','Sex_number','Age_splitter','Alone','Parch_splitter','Fare']]
labels = titanic[['Survived']]

#Splitting the data
labels=labels.astype('int')
X_train,X_test,y_train,y_test = train_test_split(train1,labels,test_size=0.2)

#Building the model
model = tree.DecisionTreeClassifier()

#Fitting the model
model.fit(X_train,y_train)

#####################################################

# Dumping the data
pickle.dump(model,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

##########################################