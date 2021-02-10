#!/usr/bin/env python
# coding: utf-8

# # BHAVANI DASARI

# # TASK1: predict the percentage of an student based on no of study hours

# In[3]:


#importing Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#READING FILES
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
dataset = pd.read_csv(url)


# # UNDERSTANDING DATASET

# In[3]:


dataset.head()
#first five cols
#for 10 dataset.head(10)


# In[4]:


dataset.tail()
#lastfive


# In[5]:


dataset.shape
#rows n cols


# In[7]:


dataset.info()


# In[8]:


dataset.describe()


# In[9]:


dataset.count


# In[12]:


dataset.isnull()
#if value is null then we get true


# In[13]:


dataset.isnull().sum()
#anyvalue is null in hours we get that coun.if one null then we get 1


# In[4]:


#Visualize Data
dataset.plot(x="Hours",y="Scores",color="red",style="*")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage")
plt.title('Hours Vs Percentage')
plt.show()


# In[20]:


dataset.corr()
#correlation means relationship between attributes.here it is 0.9 then occurence of outliers is low


# # Independent and Dependent Variables

# In[21]:


x = dataset.iloc[:,:-1].values
#all rows and cols except last
y = dataset.iloc[:,1].values
#all rows of 1st col ie scores
print(x)
print(y)


# # Splitting testing and training data

# In[23]:


#split the test data and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
#here 80% train data 20% test data


# # Training Model

# In[25]:


#Training Algorithm
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
#fiitting x,y training data to x,y axis


# In[31]:


#Visualizing Data
#y=mx+c
line = model.coef_*x+model.intercept_
#traning data with model developed
plt.scatter(x_train,y_train,color="red")
plt.plot(x,line,color='green')
plt.title('Training Data for Hrs Vs %')
plt.xlabel("HoursStudied")
plt.ylabel("Percentage")
plt.show()
#test data with model developed
plt.scatter(x_test,y_test,color="red")
plt.plot(x,line,color='green')
plt.title('Test Data for Hrs Vs %')
plt.xlabel("HoursStudied")
plt.ylabel("Percentage")
plt.show()


# # PREDICTIONS

# In[34]:


#makingprediction
#to model we are giving x_test for prediction and storing values in y_p
y_predictedValue = model.predict(x_test)
print(y_predictedValue)
print(y_test)
#compare y_p and y_t. 
#values are nearby


# In[37]:


#making prediction for our value lets take 9.25
hours = 9.25
predicted_value = model.predict([[hours]])
print(predicted_value)


# In[ ]:


#if a person studies for 9.25 hrs then 93% he obtains

