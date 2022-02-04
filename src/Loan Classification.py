#!/usr/bin/env python
# coding: utf-8

# # Loan Classification Project

# In[1]:


# Libraries we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve,recall_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV


# In[2]:


df = pd.read_csv("Dataset.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.nunique()


# - Above we can see that Reason and Bad are binary variables
# - Nothing needs to be dropped

# In[6]:


df.describe()


# In[7]:


plt.hist(df['BAD'], bins=3)
plt.show()


# In[8]:


df['LOAN'].plot(kind='density')
plt.show()


# In[9]:


plt.pie(df['REASON'].value_counts(), labels=['DebtCon', 'HomeImp'], autopct='%.1f')
plt.show()
df['REASON'].value_counts()


# In[10]:


correlation = df.corr()
sns.heatmap(correlation)
plt.show()


# In[11]:


df['BAD'].value_counts(normalize=True)


# In[12]:


df.fillna(df.mean(), inplace=True)


# In[13]:


one_hot_encoding = pd.get_dummies(df['REASON'])
df = df.drop('REASON', axis=1)
df = df.join(one_hot_encoding)
df


# In[14]:


one_hot_encoding2 = pd.get_dummies(df['JOB'])
df = df.drop('JOB', axis=1)
df = df.join(one_hot_encoding2)
df


# In[15]:


dependent = df['BAD']
independent = df.drop(['BAD'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.3, random_state=1)


# In[16]:


def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Default', 'Default'], yticklabels=['Not Default', 'Default'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


# In[17]:


dtree = DecisionTreeClassifier(class_weight={0:0.20, 1:0.80}, random_state=1)


# In[18]:


dtree.fit(x_train, y_train)


# In[19]:


dependent_performance_dt = dtree.predict(x_train)
metrics_score(y_train, dependent_performance_dt)


# - The above is perfect because we are using the train values, not the test
# - Lets test on test data

# In[20]:


dependent_test_performance_dt = dtree.predict(x_test)
metrics_score(y_test,dependent_test_performance_dt)


# - As we can see, we got decent performance from this model, lets see if we can do better
# - Selfnote: do importance features next

# In[21]:


important = dtree.feature_importances_
columns = independent.columns
important_items_df = pd.DataFrame(important, index=columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(13,13))
sns.barplot(important_items_df.Importance, important_items_df.index)
plt.show()


# - I followed this from a previous project to see the most important features
# - We can see that the most important features are DEBTINC, CLAGE and CLNO

# In[22]:


tree_estimator = DecisionTreeClassifier(class_weight={0:0.20, 1:0.80}, random_state=1)

parameters = {
    'max_depth':np.arange(2,7),
    'criterion':['gini', 'entropy'],
    'min_samples_leaf':[5,10,20,25]
             }
score = metrics.make_scorer(recall_score, pos_label=1)
gridCV= GridSearchCV(tree_estimator, parameters, scoring=score,cv=10)
gridCV = gridCV.fit(x_train, y_train) 
tree_estimator = gridCV.best_estimator_
tree_estimator.fit(x_train, y_train)


# In[23]:


dependent_performance_dt = tree_estimator.predict(x_train)
metrics_score(y_train, dependent_performance_dt)


# - We increased the less harmful error but decreased the harmful error

# In[24]:


dependent_test_performance_dt = tree_estimator.predict(x_test)
metrics_score(y_test, dependent_test_performance_dt)


# - Although the performance is slightly worse, we still reduce harmful error

# In[25]:


important = tree_estimator.feature_importances_
columns=independent.columns
importance_df=pd.DataFrame(important,index=columns,columns=['Importance']).sort_values(by='Importance',ascending=False)
plt.figure(figsize=(13,13))
sns.barplot(importance_df.Importance,importance_df.index)
plt.show()


# In[26]:


features = list(independent.columns)

plt.figure(figsize=(30,20))

tree.plot_tree(dtree,max_depth=4,feature_names=features,filled=True,fontsize=12,node_ids=True,class_names=True)
plt.show()


# - A visualization is one of the advantages that dtrees offer, we can show this to the client ot show the thought process

# In[27]:


forest_estimator = RandomForestClassifier(class_weight={0:0.20, 1:0.80}, random_state=1)
forest_estimator.fit(x_train, y_train)


# In[28]:


y_predict_training_forest = forest_estimator.predict(x_train)
metrics_score(y_train, y_predict_training_forest)


# - A perfect classification
#   - This implies overfitting

# In[29]:


y_predict_test_forest = forest_estimator.predict(x_test)
metrics_score(y_test, y_predict_test_forest)


# - The performance is a lot better than the original single tree
# - Lets fix overfitting

# In[30]:


forest_estimator_tuned = RandomForestClassifier(class_weight={0:0.20,1:0.80}, random_state=1)

parameters_rf = {  
        "n_estimators": [100,250,500],
        "min_samples_leaf": np.arange(1, 4,1),
        "max_features": [0.7,0.9,'auto'],
}

score = metrics.make_scorer(recall_score, pos_label=1)

# Run the grid search
grid_obj = GridSearchCV(forest_estimator_tuned, parameters_rf, scoring=score, cv=5)
grid_obj = grid_obj.fit(x_train, y_train)

# Set the clf to the best combination of parameters
forest_estimator_tuned = grid_obj.best_estimator_


# In[31]:


forest_estimator_tuned.fit(x_train, y_train)


# In[32]:


y_predict_train_forest_tuned = forest_estimator_tuned.predict(x_train)
metrics_score(y_train, y_predict_train_forest_tuned)


# In[33]:


y_predict_test_forest_tuned = forest_estimator_tuned.predict(x_test)
metrics_score(y_test, y_predict_test_forest_tuned)


# - We now have very good performance
# - We can submit this to the company

# ### Conclusion
# - I made many models to get the best results.
#   - The first one I made was a decision tree, this is not as good as random forest but it is transparent as it lets us visualize it. This first one had decent performance.
#   - To improve the performance of this we tried to tune the model, this reduced the harmful error.
#   - Then to improve even more I created a decision tree model, this had excellent performance once we created a second version which removed overfitting.

# ### Recommendations
# - The biggest thing that effects defaulting on a loan is the debt to income ratio. If someone has a lot of debt and a lower income they may have a harder time paying back a loan.
# - Something else that effects defaulting on a loan is the number of delinquent credit lines. This means that someone who cannot make their credit card payments will have a hard time paying back a loan.
# - Years at job is also a driver of a loans outcome. A large number of years at a job could indicate financial stability.
# - DEROG, or a history of delinquent payments is also a warning sign of not being able to pay back a loan.
# - Those are some warning signs/good signs that should be looked out for when looking for candidates to give loans to.
# 
# I will now apply SHAP to look more into this model.

# In[34]:


get_ipython().system('pip install shap')
import shap


# In[35]:


shap.initjs()


# In[36]:


explain = shap.TreeExplainer(forest_estimator_tuned)
shap_vals = explain(x_train)


# In[37]:


type(shap_vals)


# In[38]:


shap.plots.bar(shap_vals[:, :, 0])


# In[39]:


shap.plots.heatmap(shap_vals[:, :, 0])


# In[40]:


shap.summary_plot(shap_vals[:, :, 0], x_train)


# In[53]:


print(forest_estimator_tuned.predict(x_test.iloc[107].to_numpy().reshape(1,-1))) # This predicts for one row, 0 means approved, 1 means no.

