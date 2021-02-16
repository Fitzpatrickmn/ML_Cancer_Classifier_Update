#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install sklearn --upgrade
# !pip install joblib
# !pip install imbalanced-learn --upgrade
# !pip install mlxtend  

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import itertools


# In[2]:


# Read the data
nd = pd.read_csv("data.csv")
numeric_data = nd.drop(columns="Unnamed: 32")
numeric_data.head()


# In[3]:


numeric_data['diagnosis_num'] =  numeric_data['diagnosis'].apply(lambda x: 0 if x == 'B' else 1)
numeric_data.columns
numeric_data = numeric_data[['id', 'diagnosis', 'diagnosis_num', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]

numeric_data.head(15)


# In[4]:


numeric_only = numeric_data.drop('diagnosis', axis=1)
numeric_df = pd.DataFrame(numeric_only)
numeric_df.head()


# In[5]:


#
target = numeric_df['diagnosis_num']
data = numeric_df.drop('diagnosis_num', axis=1)
features = data.columns


# In[6]:


# EDA 
total_count_tumors = len(numeric_data)
print(f'Total Tumors on data = {total_count_tumors}')
total_count_benign = (numeric_data.diagnosis == 'B').sum()
print(f'Total Benign Tumors on data = {total_count_benign}')
total_count_malignant = (numeric_data.diagnosis == 'M').sum()
print(f'Total Malignant Tumors on data = {total_count_malignant}')
print("-----")
percent_benign = (total_count_benign/total_count_tumors)*100.00
percent_malignant = (total_count_malignant/total_count_tumors)*100.00
print(f"Percent Benign = {percent_benign}")
print(f"Percent Malignant = {percent_malignant}")


# # Features Selection

# Select the top 10 features according to the ```feature_selection``` from ```sklearn```

# In[7]:


bestfeatures = SelectKBest(k=31)
fit = bestfeatures.fit(data, target)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(data.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']
print(featureScores.nlargest(10,'Score'))


# In[8]:


#select the top 10 features
features_list_df = numeric_data[["concave points_worst",
"perimeter_worst",
"concave points_mean",
"radius_worst",
"perimeter_mean",
"area_worst",
"radius_mean",
"area_mean",
"concavity_mean",
"concavity_worst"]]


# ## OVERSAMPLING THE DATA

# In[9]:


# spliting the data on train and test data
X_train, X_test, y_train, y_test = train_test_split (features_list_df, target, test_size = 0.30, random_state=21)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[10]:


# Count the training data
total_count_tumors = len(y_train)
print(f'Total Tumors on train data = {total_count_tumors}')
total_count_benign = (y_train == 0).sum()
print(f'Total Benign Tumors on train data = {total_count_benign}')
total_count_malignant = (y_train == 1).sum()
print(f'Total Malignant Tumors on train data = {total_count_malignant}')
print("-----")
percent_benign = (total_count_benign/total_count_tumors)*100.00
percent_malignant = (total_count_malignant/total_count_tumors)*100.00
print(percent_benign)
print(percent_malignant)


# In[11]:


print('Original train dataset shape %s' % Counter(y_train))


# #### Apply oversampling models (RandomOverSampler, SMOTE, ADASYN)

# In[12]:


# Random Over Sampler Model
ros = RandomOverSampler(random_state=21)
X_ros, y_ros = ros.fit_resample(X_train, y_train)
print('Resampled RandomOverSampler train dataset shape %s' % Counter(y_ros))
print (X_ros.shape, y_ros.shape)


# In[13]:


# SMOTE: Synthetic Minority Oversampling Technique Model
sm = SMOTE(random_state=21)
X_sm, y_sm = sm.fit_resample(X_train, y_train)
print('Resampled SMOTE train dataset shape %s' % Counter(y_sm))
print (X_sm.shape, y_sm.shape)


# In[14]:


# ADASYN: Adaptive Synthetic Sampling Model
ada = ADASYN(random_state=21)
X_ada, y_ada = ada.fit_resample(X_train, y_train)
print('Resampled ADASYN train dataset shape %s' % Counter(y_ada))
print (X_ada.shape, y_ada.shape)


# Oversampling models comparison SMOTE VS ADASYN.
# 
# SMOTE: It finds the n-nearest neighbors in the minority class for each of the samples in the class. Then it draws a line between the the neighbors an generates random points on the lines.
# 
# ADASYN:  Works as SMOTE, but with a minor improved. After creating the sample it adds a random small values to the points. In other words instead of all the sample being linearly correlated to the parent they have a little more variance in them, been a bit scattered.
# 
# To take advantage of this smaller variation on the new data, we will use the ADASYN model to oversample the data (X_ada, y_ada).

# ## Pre-processing

# Scale the data using the ```StandardScaler```
# 
# 

# In[15]:


# Scale the data
X_scaler = StandardScaler().fit(X_ada)
X_train_scaled = X_scaler.transform(X_ada)
X_test_scaled = X_scaler.transform(X_test)
X_total_scaled = X_scaler.transform(features_list_df)


# # Machine Learn Models

# ### Model 1 - Logistic Regression

# In[16]:


# import and train the model
from sklearn import linear_model

logreg = linear_model.LogisticRegression()
logreg.fit(X_train_scaled, y_ada)
prediction_logReg = logreg.predict(X_test_scaled)


# In[17]:


print(f"Logistic Regression Training Data Score: {logreg.score(X_train_scaled, y_ada)}")
print(f"Logistic Regression Testing Data Score: {logreg.score(X_test_scaled, y_test)}\n")

print(classification_report(y_test, prediction_logReg,
                            target_names=['Benign ', 'Malignant']))


# In[18]:


# use GridSearchCV to Hyperparameter Tuning the model

param_grid_logreg = {'C': [5, 10, 50, 100],"penalty": ['l1', 'l2']}
grid_logreg = GridSearchCV(logreg, param_grid_logreg, verbose=3)
grid_logreg.fit(X_train_scaled, y_ada)


# In[19]:


print(f"Best Logistic Regression Parameters {grid_logreg.best_params_}")
print(f"Best Logistic Regression Result {grid_logreg.best_score_}")


# In[20]:


# set the best Logistic Regression model
bestLogReg_model = linear_model.LogisticRegression(C=5, penalty="l2")


# In[21]:


# apply the model 
bestLogReg_model.fit(X_train_scaled, y_ada)
print(f"Best Logistic Regression Model Testing Data Score: {bestLogReg_model.score(X_test_scaled, y_test)}")


# ### Model 2 - SVC
# 

# In[22]:


# import, train and apply the model
from sklearn.svm import SVC 

model_SVC = SVC()
model_SVC.probability = True
model_SVC.fit(X_train_scaled, y_ada)
prediction_SVC = model_SVC.predict(X_test_scaled)


# In[23]:


print(f"SVC Training Data Score: {model_SVC.score(X_train_scaled, y_ada)}")
print(f"SVC Testing Data Score: {model_SVC.score(X_test_scaled, y_test)}\n")
print(classification_report(y_test, prediction_SVC,
                            target_names=['Benign ', 'Malignant']))


# In[24]:


# use GridSearchCV to Hyperparameter Tuning the model

param_grid_SVC = {'C': [1, 5, 10, 50, 100],
              'gamma': [0.0001, 0.0005, 0.001, 0.005],
                 "kernel": ["linear",'rbf']}
grid_SVC = GridSearchCV(model_SVC, param_grid_SVC, verbose=3)


# In[25]:


grid_SVC.fit(X_train_scaled, y_ada)


# In[26]:


print(f"Best SVC Parameters {grid_SVC.best_params_}")
print(f"Best SVC Result {grid_SVC.best_score_}")


# In[27]:


# set the best SCV model
bestSVC_model = SVC(C=100, kernel='rbf', gamma=0.005)
bestSVC_model.probability = True


# In[28]:


#Best SVC model results
bestSVC_model.fit(X_train_scaled, y_ada)
print(f"Best SVC Model Testing Data Score: {bestSVC_model.score(X_test_scaled, y_test)}")


# ### Model 3 - Neural Network

# In[29]:


# import, train and apply the model
from sklearn.neural_network import MLPClassifier

nn_model = MLPClassifier()
nn_model.fit(X_train_scaled, y_ada)
prediction_nn = nn_model.predict(X_test_scaled)


# In[30]:


print(f"Neural Network Training Data Score: {nn_model.score(X_train_scaled, y_ada)}")
print(f"Neural Network Testing Data Score: {nn_model.score(X_test_scaled, y_test)}\n")

print(classification_report(y_test, prediction_nn,
                            target_names=['Benign ', 'Malignant']))


# In[31]:


# use GridSearchCV to Hyperparameter Tuning the model

param_grid_nn = {'hidden_layer_sizes': [1,3,5,9,13,18,20],
              "activation":["identity", "logistic", "tanh", "relu"],
                 "solver":["lbfgs", "sgd", "adam"],
                }
grid_nn = GridSearchCV(nn_model, param_grid_nn, verbose=3,cv=3,
                           scoring='accuracy')


# In[32]:


grid_nn.fit(X_train_scaled, y_ada)


# In[33]:


print(f"Best Neural Network Parameters {grid_nn.best_params_}")
print(f"Best Neural Network Result {grid_nn.best_score_}")


# In[34]:


# set the best Neural Network model
bestnn_model = MLPClassifier(activation= 'relu', hidden_layer_sizes= 20, solver= 'lbfgs')


# In[35]:


#Best Neural Network model results
bestnn_model.fit(X_train_scaled, y_ada)
print(f"Best Neural Network Model Testing Data Score: {bestnn_model.score(X_test_scaled, y_test)}")


# ### Model 4 - Random Forest

# In[36]:


# import the model 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_ada)
prediction_rf = rf.predict(X_test_scaled)


# In[37]:


print(f"RandonForest Training Data Score: {rf.score(X_train_scaled, y_ada)}")
print(f"RandonForest Testing Data Score: {rf.score(X_test_scaled, y_test)}\n")

print(classification_report(y_test, prediction_rf,
                            target_names=['Benign ', 'Malignant']))


# In[38]:


# use GridSearchCV to Hyperparameter Tuning the model

param_grid_rf = {'n_estimators': [250, 300, 350]
                 ,'max_depth': [125, 150, 175]}
grid_rf = GridSearchCV(rf, param_grid_rf, verbose=3)


# In[39]:


grid_rf.fit(X_train_scaled, y_ada)


# In[40]:


print(f"Best RandomForest Parameters {grid_rf.best_params_}")
print(f"Best RandomForest result {grid_rf.best_score_}")


# In[41]:


# set the best RandomForest model
best_rf = RandomForestClassifier(max_depth=150, n_estimators=350)
best_rf.probability = True


# In[42]:


best_rf.fit(X_train_scaled, y_ada)
print(f"Best RandomForest Model Testing Data Score: {best_rf.score(X_test_scaled, y_test)}")


# ### Model 5 - Gradient Boost

# In[43]:


# import the model 
from sklearn.ensemble import GradientBoostingClassifier

grad_clf = GradientBoostingClassifier()
grad_clf.fit(X_train_scaled, y_ada)
prediction_grad = grad_clf.predict(X_test_scaled)


# In[44]:


print(f"GradientBoosting Training Data Score: {grad_clf.score(X_train_scaled, y_ada)}")
print(f"GradientBoosting Testing Data Score: {grad_clf.score(X_test_scaled, y_test)}")

print(classification_report(y_test, prediction_grad,
                            target_names=['Benign ', 'Malignant']))


# In[45]:


# use GridSearchCV to Hyperparameter Tuning the model

param_grid_grd = {'n_estimators': [100, 250, 300, 350],'max_depth': [3, 50, 100, 150, 200]}
grid_grad = GridSearchCV(grad_clf, param_grid_grd, verbose=3)
grid_grad.fit(X_train_scaled, y_ada)


# In[46]:


print(f"Best GradientBoosting Parameters {grid_grad.best_params_}")
print(f"Best GradientBoosting result {grid_grad.best_score_}")


# In[47]:


# set the best GradientBoosting model
bestgrad_model = GradientBoostingClassifier(max_depth=3,n_estimators=350)


# In[48]:


#Best SVC model results
bestgrad_model.fit(X_train_scaled, y_ada)
print(f"Best GradientBoosting Model Testing Data Score: {bestgrad_model.score(X_test_scaled, y_test)}")


# ## Ensemble a better Model

# ### Test the tuned models accuracies with ```crossvalidation```

# In[49]:


from sklearn import model_selection

clf1 = bestLogReg_model
clf2 = bestSVC_model
clf3 = bestnn_model
clf4 = best_rf
clf5 = bestgrad_model


labels = ["Logistic Regression", 'SVC', "Neural Network",'Random Forest', 'GradientBoosting']

for clf, label in zip([clf1, clf2, clf3, clf4, clf5], labels):

        scores = model_selection.cross_val_score(clf, X_test_scaled, y_test, scoring='accuracy')
        predicted = model_selection.cross_val_predict(clf,X_test_scaled, y_test)
        diff = predicted - y_test
        misclass_indexes = diff[diff != 0].index.tolist()
        print(f"{label} Model - Accuracy: {scores.mean():0.3f} (+/- {scores.std():0.3f})\nMissclassed data index {misclass_indexes}\n")


# ### Combining the models to ensemble an improved model

# #### Giving different weight to models based on least missclassed data

# In[52]:



from mlxtend.classifier import EnsembleVoteClassifier
import copy
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3, clf4, clf5], weights=[2,2,2,2,1], refit=False)
eclf.fit(X_test_scaled, y_test)

print('Ensemble Model accuracy:', np.mean(y_test == eclf.predict(X_test_scaled))*100, "%")


# ## The Final ensemble model gave us an accuracy of 95.90%.

# In[ ]:




