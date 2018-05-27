
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# In[2]:


for dataset in [train_data, test_data]:
    dataset['Year'] = dataset['Dates'].apply(lambda x:int(x.split('-')[0]))
    dataset['Month'] = dataset['Dates'].apply(lambda x:int(x.split('-')[1]))
    dataset['Date'] = dataset['Dates'].apply(lambda x:int(x.split('-')[2].split(' ')[0]))
    dataset['Hour'] = dataset['Dates'].apply(lambda x:int(x.split(' ')[1].split(':')[0]))
    dataset['Minute'] = dataset['Dates'].apply(lambda x:int(x.split(' ')[1].split(':')[1]))


# In[3]:


threat_to_individuals = ['ARSON', 'ASSAULT', 'EXTORTION', 'KIDNAPPING', 'LARCENY/THEFT', 'BURGLARY', 'MISSING PERSON', 'ROBBERY', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'VANDALISM', 'VEHICLE THEFT', 'WEAPON LAWS', 'FAMILY OFFENSES', 'OTHER OFFENSES']
violation_of_law = ['BAD CHECKS', 'BRIBERY', 'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'LIQUOR LAWS', 'LOITERING', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'RUNAWAY', 'SECONDARY CODES', 'SUSPICIOUS OCC', 'TRESPASS', 'WARRANTS']

train_data['Category'] = train_data['Category'].apply(lambda x: 'THREAT TO OTHERS LIFE' if x in threat_to_individuals else ('VIOLATION OF LAW' if x in violation_of_law  else 'NON-CRIMINAL'))


# In[4]:


train_data = train_data.fillna(train_data.mode().iloc[0])
test_data = test_data.fillna(test_data.mode().iloc[0])


# In[8]:


features = ['Year','Month','Date','Hour','Minute','DayOfWeek','PdDistrict','X','Y']
non_numeric_features = ['DayOfWeek', 'PdDistrict']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)

for feature in non_numeric_features:
    labelEncodedData = le.fit_transform(list(train_data[feature]) + list(test_data[feature]))
    oneHotEncoderData = labelEncodedData.reshape(len(labelEncodedData), 1)
    ohe.fit(oneHotEncoderData)
    train_data = train_data.combine_first(pd.DataFrame(ohe.transform(le.transform(train_data[feature]).reshape(len(train_data[feature]),1))))
    test_data = test_data.combine_first(pd.DataFrame(ohe.transform(le.transform(test_data[feature]).reshape(len(test_data[feature]),1))))


# In[9]:


features = ['Year','Month','Date','Hour','Minute','X','Y','0','1','2','3','4','5','6','7','8','9']
train_data.columns = ['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y','Year',
                      'Month','Date','Hour','Minute','0','1','2','3','4','5','6','7','8','9']
print(train_data.columns.values)
train_data.head()


# In[10]:


from sklearn.model_selection import train_test_split
train_set, valid_set, train_labels, valid_labels = train_test_split(train_data[list(features)], train_data['Category'], test_size=0.4, random_state=4327)


# In[11]:


from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

classifiers = [
        RandomForestClassifier(max_depth=16,n_estimators=1024),
        GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=5, random_state=0),
        KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=100, p=10, metric='minkowski'),
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=8), algorithm="SAMME.R", n_estimators=128)
    ]


# In[12]:


for classifier in classifiers:
    # Train
    classifier.fit(train_set, train_labels)

    # Test results
    print(classifier.__class__.__name__)
    print('Accuracy Score:')
    print(accuracy_score(valid_labels,classifier.predict(valid_set)))


# In[47]:


from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(train_set, train_labels)
print(accuracy_score(valid_labels, xgb_classifier.predict(valid_set)))

