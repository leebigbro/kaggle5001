from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns
import datetime as dt
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from time import time
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error


df=pd.read_csv("train.csv")
df = df.set_index("id")
te = pd.read_csv("test.csv")
te = te.set_index("id")

df.dropna(how='any',inplace=True)
df=pd.concat([df,te],axis = 0)

te = pd.read_csv("test.csv")
te = te.set_index("id")

#review
df['ratio']=(df['total_positive_reviews']+1)/(df['total_negative_reviews']+1)

#date
df['purchase_date']=pd.to_datetime(df['purchase_date'])
df['purchase_year']=df['purchase_date'].apply(lambda x:x.year)
df['purchase_month']=df['purchase_date'].apply(lambda x:x.month)


df['release_date']=pd.to_datetime(df['release_date'])
df['release_year']=df['release_date'].apply(lambda x:x.year)
df['release_month']=df['release_date'].apply(lambda x:x.month)

today=pd.to_datetime(dt.datetime.today().strftime("%m/%d/%Y"))
df['pur_date_dis']=(today-df['purchase_date']).dt.days

df['date_dis']=(df['purchase_date']-df['release_date']).dt.days
df['date_dis0']=[1 if c>0 else 0 for c in df['date_dis']]
df.drop('date_dis', axis=1,inplace=True)

df.drop(['purchase_date','release_date'], axis=1,inplace=True)


d3=pd.get_dummies(df['purchase_year'])
df = pd.concat([df,d3],axis = 1)

df.drop(['purchase_year','release_year','purchase_month','release_month'], axis=1,inplace=True)

df1 = df['genres'].str.get_dummies(",")
df2 = df['categories'].str.get_dummies(",")
df3 = df['tags'].str.get_dummies(",")
# display 

df1_1=df1[['Simulation','Strategy','RPG','Action','Adventure']]


df2_1=df2[['Steam Leaderboards','Steam Workshop']]



df.drop(['genres','categories','tags'], axis = 1, inplace=True)


df = pd.concat([df,df1],axis = 1)

t=df.shape[0]-90
test=df.iloc[t:,]
train =df.iloc[:t,]
X_train =train.drop(["playtime_forever"], axis=1)
y_train= train["playtime_forever"]
test=test.drop(["playtime_forever"], axis=1)

random = RandomForestRegressor()

n_estimators = [int(x) for x in np.linspace(start = 50, stop = 600, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(4, 20, num = 5)]
min_samples_split = [2, 3,4,5,10, 15,20]
min_samples_leaf = [1, 2,3,5,10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
pprint(random_grid)


random = RandomForestRegressor(oob_score=True,criterion='mse')
print('Parameters currently in use:\n')
pprint(random.get_params())
rf_random = RandomizedSearchCV(estimator = random,
                               param_distributions = random_grid,
                               n_iter = 100, cv = 3, verbose=2,
                               random_state=42, n_jobs = -1)
rf_random.fit(X_train,y_train)
random=rf_random.best_estimator_
y_pred =random.predict(test)
result=pd.DataFrame(data=y_pred)
result.to_csv("test.csv", index=True)