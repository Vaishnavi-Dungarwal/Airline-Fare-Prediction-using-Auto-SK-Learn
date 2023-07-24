import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("/content/drive/MyDrive/flight data.csv")

df.head()

df.shape

df1= pd.read_csv("/content/drive/MyDrive/flight test.csv")

df1.head()

df1.shape

total_df= df.append(df1,sort= False)

total_df.tail()

total_df.head()

total_df.shape

"""#### Size of training set: 10683 records

#### Size of test set: 2671 records

### FEATURES:

#### Airline: The name of the airline.

#### Date_of_Journey: The date of the journey

#### Source: The source from which the service begins.

#### Destination: The destination where the service ends.

#### Route: The route taken by the flight to reach the destination.

#### Dep_Time: The time when the journey starts from the source.

#### Arrival_Time: Time of arrival at the destination.

#### Duration: Total duration of the flight.

#### Total_Stops: Total stops between the source and destination.

#### Additional_Info: Additional information about the flight

#### Price: The price of the ticket
"""

df['Airline'].value_counts() ## checking for airline data

sns.catplot(y='Price',x='Airline',data= df.sort_values('Price',ascending=False),kind="boxen",height=6, aspect=3)
plt.show

df['Source'].value_counts() ## checking for Source

sns.catplot(y='Price',x='Source',data= df.sort_values('Price',ascending=False),kind="boxen",height=6, aspect=3)
plt.show

df['Destination'].value_counts() ## checking for Destination

sns.catplot(y='Price',x='Destination',data= df.sort_values('Price',ascending=False),kind="boxen",height=6, aspect=3)
plt.show

df['Duration'].value_counts()

total_df['Date']=total_df['Date_of_Journey'].str.split('/').str[0]
total_df['Month']=total_df['Date_of_Journey'].str.split('/').str[1]
total_df['Year']=total_df['Date_of_Journey'].str.split('/').str[2]

total_df.head()

total_df=total_df.drop(['Date_of_Journey'],axis=1)

total_df.dtypes

total_df['Date']=total_df['Date'].astype(int)
total_df['Month']=total_df['Month'].astype(int)
total_df['Year']=total_df['Year'].astype(int)

total_df.dtypes

total_df.head()

total_df['Additional_Info'].value_counts()

sns.catplot(y='Price',x='Additional_Info',data= df.sort_values('Price',ascending=False),kind="boxen",height=6, aspect=3)
plt.show

total_df['Arrival_Time']=total_df['Arrival_Time'].str.split(' ').str[0]

total_df.head()

total_df['Total_Stops'].value_counts()

total_df['Total_Stops']=total_df['Total_Stops'].fillna('1 stop')

total_df['Total_Stops'].isnull().sum()

total_df['Total_Stops']=total_df['Total_Stops'].replace('non-stop','0 stop')

total_df.head()

total_df['Stop'] = total_df['Total_Stops'].str.split(' ').str[0]

total_df.head()

total_df=total_df.drop(['Total_Stops'],axis=1)

total_df.head()

total_df.dtypes

total_df['Stop']=total_df['Stop'].astype(int)

total_df.dtypes

total_df['Arrival_Hour'] = total_df['Arrival_Time'] .str.split(':').str[0]
total_df['Arrival_Minute'] = total_df['Arrival_Time'] .str.split(':').str[1]

total_df['Arrival_Hour']=total_df['Arrival_Hour'].astype(int)
total_df['Arrival_Minute']=total_df['Arrival_Minute'].astype(int)
total_df=total_df.drop(['Arrival_Time'],axis=1)

total_df.head()

total_df['Departure_Hour'] = total_df['Dep_Time'] .str.split(':').str[0]
total_df['Departure_Minute'] = total_df['Dep_Time'] .str.split(':').str[1]

total_df['Departure_Hour']=total_df['Departure_Hour'].astype(int)
total_df['Departure_Minute']=total_df['Departure_Minute'].astype(int)
total_df=total_df.drop(['Dep_Time'],axis=1)

total_df.head()

total_df['Route_1']= total_df['Route'].str.split('→ ').str[0]
total_df['Route_2']= total_df['Route'].str.split('→ ').str[1]
total_df['Route_3']= total_df['Route'].str.split('→ ').str[2]
total_df['Route_4']= total_df['Route'].str.split('→ ').str[3]
total_df['Route_5']= total_df['Route'].str.split('→ ').str[4]

total_df['Route_1'].fillna("None",inplace=True)
total_df['Route_2'].fillna("None",inplace=True)
total_df['Route_3'].fillna("None",inplace=True)
total_df['Route_4'].fillna("None",inplace=True)
total_df['Route_5'].fillna("None",inplace=True)

total_df.head()

total_df=total_df.drop(['Route'],axis=1)
total_df=total_df.drop(['Duration'],axis=1)

total_df.head()

total_df.isnull().sum()

total_df['Price'].fillna((total_df['Price'].mean()),inplace=True)

total_df.isnull().sum()

total_df.head()

from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()

total_df["Airline"]=le.fit_transform(total_df['Airline'])
total_df["Source"]=le.fit_transform(total_df['Source'])
total_df["Destination"]=le.fit_transform(total_df['Destination'])
total_df["Additional_Info"]=le.fit_transform(total_df['Additional_Info'])
total_df["Route_1"]=le.fit_transform(total_df['Route_1'])
total_df["Route_2"]=le.fit_transform(total_df['Route_2'])
total_df["Route_3"]=le.fit_transform(total_df['Route_3'])
total_df["Route_4"]=le.fit_transform(total_df['Route_4'])
total_df["Route_5"]=le.fit_transform(total_df['Route_5'])

total_df.head()

total_df= total_df.drop(['Year'], axis=1)

total_df.head()

x= total_df.drop(['Price'],axis=1)
y= total_df['Price']

x.head()

y.head()

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

model=SelectFromModel(Lasso(alpha=0.005,random_state=0))

model.fit(x_train,y_train)

model.get_support()

selected_features=x_train.columns[(model.get_support())]

selected_features

from sklearn.ensemble import ExtraTreesRegressor

etr = ExtraTreesRegressor()
etr.fit(x_train,y_train)

plt.style.use("ggplot")
plt.figure(figsize = (15,8))
feature_importances = pd.Series(etr.feature_importances_, index=x.columns)
feature_importances.nlargest(35).plot(kind='bar')
plt.show()

from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(x_train, y_train)

y_pred = reg_rf.predict(x_test)

reg_rf.score(x_train, y_train)

reg_rf.score(x_test, y_test)

plt.style.use("ggplot")
plt.figure(figsize=(12,8))
sns.distplot(y_test-y_pred)
plt.show()

plt.figure(figsize=(12,8))
plt.style.use("ggplot")
plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

metrics.r2_score(y_test, y_pred)

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1500, num = 15)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 5, 7, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2,3, 5,8, 10]

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = param_grid,scoring='neg_mean_squared_error', n_iter = 50, cv = 5,
                               verbose=2, random_state=42, n_jobs = 3)

rf_random.fit(x_train,y_train)

rf_random.best_params_

reg_rf = RandomForestRegressor(n_estimators= 700,
 min_samples_split=3,
 min_samples_leaf= 1,
 max_features='sqrt',
 max_depth= 15)

reg_rf.fit(x_train,y_train)

ypred1= reg_rf.predict(x_test)

ypred1

plt.style.use("ggplot")
plt.figure(figsize=(12,8))
sns.distplot(y_test-ypred1)
plt.show()

plt.figure(figsize=(12,8))
plt.style.use("ggplot")
plt.scatter(y_test, ypred1, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_predq")
plt.show()

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, ypred1))
print('MSE:', metrics.mean_squared_error(y_test, ypred1))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ypred1)))

metrics.r2_score(y_test, ypred1)

!apt-get install swig -y
!pip install Cython numpy
!pip install auto-sklearn

import autosklearn

import autosklearn.regression as reg

from google.colab import drive
drive.mount('/content/drive')

df= pd.read_csv("/content/drive/MyDrive/flight data.csv")

df.head()

df1= pd.read_csv("/content/drive/MyDrive/flight test.csv")

"""We will use the same x_train and y_train data"""

x_train

automl = reg.AutoSklearnRegressor(time_left_for_this_task=120,per_run_time_limit=30)

automl.fit(x_train,y_train)

print(automl.leaderboard())

print(automl.show_models())

ypred2= automl.predict(x_test)

ypred2

plt.style.use("ggplot")
plt.figure(figsize=(12,8))
sns.distplot(y_test-ypred2)
plt.show()

plt.figure(figsize=(12,8))
plt.style.use("ggplot")
plt.scatter(y_test, ypred2, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("ypred2")
plt.show()

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, ypred2))
print('MSE:', metrics.mean_squared_error(y_test, ypred2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ypred2)))

