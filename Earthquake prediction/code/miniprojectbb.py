#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv("C:\\Users\\Vilok Sai Boddapati\\Downloads\\dataset.csv")
print(data)


# In[3]:


data.corr()


# In[4]:


data.info()


# In[5]:


data.isnull()


# In[6]:


data.isnull().sum()


# In[7]:


X=data[['Latitude','Longitude','Depth']]
Y=data['Magnitude']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)


# In[8]:


#multi linear regression
from sklearn.linear_model import LinearRegression
mlr=LinearRegression()
mlr.fit(x_train,y_train)
print("Intercept:",mlr.intercept_)
print("coefficients:")
list(zip(X,mlr.coef_))


# In[11]:


y_pred_mlr=mlr.predict(x_test)
#print(y_pred_mlr)


# In[42]:


from sklearn.metrics import mean_squared_error
rmse_mlr=np.sqrt(mean_squared_error(y_test,y_pred_mlr))
print(rmse_mlr)


# In[64]:


from sklearn.metrics import r2_score
r2s_mlr=r2_score(y_test,y_pred_mlr)
print(r2s_mlr)


# In[65]:


from sklearn.metrics import mean_absolute_error as mae
mae_mlr=mae(y_test,y_pred_mlr)
print(mae_mlr)


# In[45]:


#support vector
from sklearn import svm
sv=svm.SVR()
sv.fit(x_train,y_train)
y_pred_svr=sv.predict(x_test)


# In[46]:


from sklearn.metrics import mean_squared_error
rmse_svr=np.sqrt(mean_squared_error(y_test,y_pred_svr))
print(rmse_svr)


# In[66]:


from sklearn.metrics import r2_score
r2s_svr=r2_score(y_test,y_pred_svr)
print(r2s_svr)


# In[67]:


from sklearn.metrics import mean_absolute_error as mae
mae_svr=mae(y_test,y_pred_svr)
print(mae_svr)


# In[49]:


#random forest regression
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators = 100, random_state = 0)
rfr.fit(x_train,y_train)
y_pred_rfr=rfr.predict(x_test)


# In[50]:


from sklearn.metrics import mean_squared_error
rmse_rfr=np.sqrt(mean_squared_error(y_test,y_pred_rfr))
print(rmse_rfr)


# In[68]:


from sklearn.metrics import r2_score
r2s_rfr=r2_score(y_test,y_pred_rfr)
print(r2s_rfr)


# In[69]:


from sklearn.metrics import mean_absolute_error as mae
mae_rfr=mae(y_test,y_pred_rfr)
print(mae_rfr)


# In[53]:


#decisiontreeregression
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state = 0)
dtr.fit(x_train,y_train)
y_pred_dtr=dtr.predict(x_test)


# In[54]:


from sklearn.metrics import mean_squared_error
rmse_dtr=np.sqrt(mean_squared_error(y_test,y_pred_dtr))
print(rmse_dtr)


# In[70]:


from sklearn.metrics import r2_score
r2s_dtr=r2_score(y_test,y_pred_dtr)
print(r2s_dtr)


# In[71]:


from sklearn.metrics import mean_absolute_error as mae
mae_dtr=mae(y_test,y_pred_dtr)
print(mae_dtr)


# In[57]:


#lasso regression
from sklearn.linear_model import Lasso
las = Lasso(alpha=1)
las.fit(x_train, y_train)
y_pred_las=las.predict(x_test)


# In[58]:


from sklearn.metrics import mean_squared_error
rmse_las=np.sqrt(mean_squared_error(y_test,y_pred_las))
print(rmse_las)


# In[72]:


from sklearn.metrics import r2_score
r2s_las=r2_score(y_test,y_pred_las)
print(r2s_las)


# In[73]:


from sklearn.metrics import mean_absolute_error as mae
mae_las=mae(y_test,y_pred_las)
print(mae_las)


# In[63]:


scores = [rmse_mlr,rmse_svr,rmse_rfr,rmse_dtr,rmse_las]
algorithms = ["Multi-Linear","Support Vector","Random Forest","Decision Tree","Laso "]    
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(8,5)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(algorithms,scores)


# In[74]:


scores = [r2s_mlr,r2s_svr,r2s_rfr,r2s_dtr,r2s_las]
algorithms = ["Multi-Linear","Support Vector","Random Forest","Decision Tree","Laso "]    
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(8,5)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(algorithms,scores)


# In[75]:


scores = [mae_mlr,mae_svr,mae_rfr,mae_dtr,mae_las]
algorithms = ["Multi-Linear","Support Vector","Random Forest","Decision Tree","Laso "]    
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(8,5)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(algorithms,scores)


# In[77]:


X=data.iloc[:,:-1]
Y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(X,Y,random_state=100)
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(x_train,y_train)
y_pred=rf.predict([[18,94,5]])
print(y_pred)
if(y_pred<3.7):
    print("no chance for earthquake")
else:
    print("earthquake can occur")


# In[ ]:




