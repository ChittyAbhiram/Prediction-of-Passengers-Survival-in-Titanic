#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[4]:


import pandas as pd


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


import seaborn as sns


# In[7]:


tit = sns.load_dataset('titanic')


# In[8]:


tit


# In[9]:


tit.head(10)


# In[10]:


tit.shape


# In[11]:


tit['survived'].value_counts()


# In[12]:


tit.describe


# In[13]:


sns.countplot(tit['survived'],label="Count")


# In[14]:


tit.groupby('sex')[['survived']].mean()


# In[15]:


tit.pivot_table('survived', index='sex', columns='class')


# In[16]:


tit.pivot_table('survived', index='sex', columns='class').plot()


# In[18]:


# Visualizing the count of survivors for columns 'who', 'sex', 'pclass', 'sibsp', 'parch', and 'embarked'
cols = ['who', 'sex', 'pclass', 'sibsp', 'parch', 'embarked']

n_rows = 2
n_cols = 3

# The subplot grid and the figure size of each graph
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.2,n_rows*3.2))

for r in range(0,n_rows):
    for c in range(0,n_cols):  
        
        i = r*n_cols+ c #index to go through the number of columns       
        ax = axs[r][c] #Show where to position each subplot
        sns.countplot(tit[cols[i]], hue=tit["survived"], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title="survived", loc='upper right') 
        
plt.tight_layout()   #tight_layout


# In[23]:


sns.barplot(x='class', y='survived', data=tit)


# In[24]:


age = pd.cut(tit['age'], [0, 18, 80])
tit.pivot_table('survived', ['sex', age], 'class')


# In[38]:


#the prices paid for each class.
x = tit['fare'].head(20)
y = tit['class'].head(20)
plt.scatter(x,y)


# In[39]:


tit.isna().sum()


# In[40]:


for val in tit:
    print(tit[val].value_counts())
    print()


# In[41]:


tit = tit.drop(['deck', 'embark_town', 'alive', 'class', 'alone', 'adult_male', 'who'], axis=1)

#Remove the rows with missing values
tit = tit.dropna(subset =['embarked', 'age'])


# In[43]:


tit.shape


# In[44]:


print(tit['sex'].unique())
print(tit['embarked'].unique())


# In[45]:


#Converting alphabets into numerals
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#Encode sex column
tit.iloc[:,2]= labelencoder.fit_transform(tit.iloc[:,2].values)
#print(labelencoder.fit_transform(titanic.iloc[:,2].values))

#Encode embarked
tit.iloc[:,7]= labelencoder.fit_transform(tit.iloc[:,7].values)
#print(labelencoder.fit_transform(titanic.iloc[:,7].values))

#Print the NEW unique values in the columns
print(tit['sex'].unique())
print(tit['embarked'].unique())


# In[46]:


#Split the data into independent ‘X’ and dependent ‘Y’ data sets.
X = tit.iloc[:, 1:8].values 
Y = tit.iloc[:, 0].values 


# In[47]:


# Split the dataset into 80% Training set and 20% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[48]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[49]:


#Create a function within many Machine Learning Models
def models(X_train,Y_train):
    
  
  #Using Logistic Regression Algorithm to the Training Set
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, Y_train)
  
  #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    knn.fit(X_train, Y_train)

  #Using SVC method of svm class to use Support Vector Machine Algorithm
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X_train, Y_train)

  #Using SVC method of svm class to use Kernel SVM Algorithm
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train, Y_train)

  #Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)

  #Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, Y_train)

  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)
  
  #print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
    print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
    print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  
    return log, knn, svc_lin, svc_rbf, gauss, tree, forest


# In[50]:


#Get and train all of the models
model = models(X_train,Y_train)


# In[51]:


from sklearn.metrics import confusion_matrix 
for i in range(len(model)):
    cm = confusion_matrix(Y_test, model[i].predict(X_test)) 
   #extracting TN, FP, FN, TP
    TN, FP, FN, TP = confusion_matrix(Y_test, model[i].predict(X_test)).ravel()
    print(cm)
    print('Model[{}] Testing Accuracy = "{} !"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))
    print()# Print a new line


# In[52]:


#Get the importance of the features
forest = model[6]
importances = pd.DataFrame({'feature':tit.iloc[:, 1:8].columns,'importance':np.round(forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances


# In[53]:


importances.plot.bar()


# In[57]:


#Print Prediction using Random Forest Classifier, ‘1’ means the passenger survived and ‘0’ means the passenger did not survive.
pred = model[6].predict(X_test)
print(pred)

#Print a space
print()

#Print the actual values
print(Y_test)


# # Result - Random Forest Classifier(Model 6) is selected as the best model to predict survivors because it has an accuracy of 80.41% on the testing data and 97.53% on the training data.

# In[ ]:




