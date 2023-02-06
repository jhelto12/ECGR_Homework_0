#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Name - Jordan Helton
Student ID - 801012514
Homework 0
Github Repo - https://github.com/jhelto12/ECGR_Homework_0

"""


# In[ ]:


"""

Problem 1
    Part 1 - See Below
    Part 2 - See Below
    Part 3 - [Column] had the lowest loss or cost because it had the highest slope 
    Part 4 - A higher learning rate means a higher slope of the graph and a lower cost
    
"""


# In[80]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[81]:


dataset = pd.read_csv('https://raw.githubusercontent.com/jhelto12/ECGR_Homework_0/main/D3.csv')
dataset.head()
M = len(dataset)


# In[82]:


x1Column = dataset.values[:, 0]
x2Column = dataset.values[:, 1]
x3Column = dataset.values[:, 2]
yColumn = dataset.values[:, 3]
exampleCnt = len(yColumn)

x1 = x1Column
x2 = x2Column
x3 = x3Column

print('X1 = ', x1Column[: 5])
print('X2 = ', x2Column[: 5])
print('X3 = ', x3Column[: 5])
print('Y = ', yColumn[: 5])
print('Number of Training Examples = ', exampleCnt)


# In[83]:


#function for calculating the Cost
def computeCost(inColumn, outColumn, theta):
    
    prediction = inColumn.dot(theta)
    error = np.subtract(prediction, outColumn)
    squareError = np.square(error)
    
    J = 1/(2*exampleCount) * np.sum(squareError)
    
    return J


# In[84]:


#function for calculating Gradient Decent
def gradientDecent(inColumn, outColumn, theta, alpha, iterations):
    costHist = np.zeros(iterations)
    
    for i in range(iterations):
        prediction = inColumn.dot(theta)
        error = np.subtract(prediction, outColumn)
        deltaSum = (alpha/exampleCount) * inColumn.transpose().dot(error)
        theta = theta - deltaSum
        costHist[i] = computeCost(inColumn, outColumn, theta)
        
    return theta, costHist


# In[106]:


#function for setting up the column graph displays
def graphDisplay(inColumn, outColumn, inputColor):
    plt.scatter(inColumn, outColumn, color = inputColor)
    plt.rcParams["figure.figsize"] = (10,6)
    plt.grid()
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Gradient Decent Fit')


# In[107]:


#Column Graphs
graphDisplay(x1Column, yColumn, 'red')
graphDisplay(x2Column, yColumn, 'green')
graphDisplay(x3Column, yColumn, 'blue')


# In[105]:


Col1 = np.ones((exampleCnt, 1))
Col2 = x1.reshape(exampleCnt, 1)
x2 = np.hstack((Col1, Col2))
alphaVal = 0.001 #0.01
iterationCnt = 1500
thetaVal = np.zeros(2)
costVal = computeCost(x2, yColumn, thetaVal)
thetaVal, costHist = gradientDecent(x2, yColumn, thetaVal, alphaVal, iterationCnt)

graphDisplay(x1Column, yColumn, thetaVal)
plt.plot(x1Column, x2.dot(thetaVal), color = 'red', label = 'X2 Linear Regression')
plt.title('X1 Scatter Plot')


# In[102]:


plt.plot(range(1, iterationCnt+1), costHist, color = 'red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Loss (J)')
plt.ylabel('X1 Gradient Decent')


# In[ ]:




