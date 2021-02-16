#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install sklearn --upgrade')
get_ipython().system('pip install joblib')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


nd = pd.read_csv("numeric_data.csv")
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

numeric_data.head()


# In[4]:


numeric_only = numeric_data.drop('diagnosis', axis=1)


# In[5]:


numeric_df = pd.DataFrame(numeric_only)
numeric_df.head()


# In[6]:


target = numeric_df['diagnosis_num']
data = numeric_df.drop('diagnosis_num', axis=1)
features = data.columns


# In[7]:


# EDA 
total_count_tumors = len(numeric_data)
print(total_count_tumors)
total_count_benign = (numeric_data.diagnosis == 'B').sum()
print(total_count_benign)
total_count_malignant = (numeric_data.diagnosis == 'M').sum()
print(total_count_malignant)
print("-----")
percent_benign = (total_count_benign/total_count_tumors)*100.00
percent_malignant = (total_count_malignant/total_count_tumors)*100.00
print(percent_benign)
print(percent_malignant)


# In[8]:


data.head()


# In[9]:


features.shape


# In[10]:


#Check for Null Values
numeric_df.isnull().sum().sort_values()


# In[34]:


# Feature Key
# se = standard_error; worst = "worst" or largest mean value; 

from sklearn.feature_selection import SelectKBest

bestfeatures = SelectKBest(k=31)
fit = bestfeatures.fit(data, target)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(data.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']
print(featureScores.nlargest(31,'Score')) 


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

corrmat = numeric_df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(numeric_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#Features that don't correlate: 
#ID, texture_mean, texture_se, smoothness_se, fractal_dimension_se, fractal_dimension_mean 


# In[33]:


#Visualizing Data from Top Features
top = featureScores.nlargest(10,'Score')
top


# In[14]:


# Data and Feature Exploration

plt.hist(numeric_df['perimeter_worst'])
plt.xlabel("Value")
plt.ylabel("Frequency")


# In[15]:


ax1 = numeric_df.plot.scatter(x='perimeter_worst',
                      y='perimeter_mean',
                      c='diagnosis_num',
                      colormap='viridis')


# In[16]:


plt.hist(numeric_df['area_mean'])
plt.xlabel("Value")
plt.ylabel("Frequency")


# In[17]:


ax2 = numeric_df.plot.scatter(x='area_worst',
                      y='area_mean',
                      c='diagnosis_num',
                      colormap='viridis')


# In[18]:


plt.hist(numeric_df['concave points_worst'])
plt.xlabel("Value")
plt.ylabel("Frequency")


# In[19]:


plt.hist(numeric_df['concave points_mean'])
plt.xlabel("Value")
plt.ylabel("Frequency")


# In[20]:


ax3 = numeric_df.plot.scatter(x='concave points_worst',
                      y='concave points_mean',
                      c='diagnosis_num',
                      colormap='viridis')


# In[21]:


plt.hist(numeric_df['radius_mean'])


# In[22]:


plt.plot(numeric_df['radius_mean'])

ax4 = numeric_df.plot.scatter(x='radius_worst',
                      y='radius_mean',
                      c='diagnosis_num',
                      colormap='viridis')


# In[23]:


plt.hist(numeric_data['symmetry_mean'])
plt.xlabel("Value")
plt.ylabel("Frequency")


# In[24]:


ax5 = numeric_df.plot.scatter(x='symmetry_worst',
                      y='symmetry_mean',
                      c='diagnosis_num',
                      colormap='viridis')


# In[25]:


index = list(numeric_df.index)


# In[26]:


x = np.arange(len(numeric_df))
xs = pd.Series(x)


# In[27]:


cpw = numeric_df.plot.scatter(x='diagnosis_num',
                      y='concave points_worst',
                      c='diagnosis_num',
                      colormap='viridis')


# In[29]:


pw = numeric_df.plot.scatter(x='diagnosis_num',
                      y='perimeter_worst',
                      c='diagnosis_num',
                      colormap='viridis')


# In[30]:


cpm = numeric_df.plot.scatter(x='diagnosis_num',
                      y='concave points_mean',
                      c='diagnosis_num',
                      colormap='viridis')


# In[ ]:




