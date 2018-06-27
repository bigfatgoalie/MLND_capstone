
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb

color = sns.color_palette()
get_ipython().magic('matplotlib inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment=None
pd.options.display.max_columns=9999


# In[2]:


train_df = pd.read_csv('train.csv')


# In[3]:


train_df.head()


# In[4]:


temp_cols = ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]

#humidity sensor columns
rho_cols = ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_6","RH_7","RH_8","RH_9"]

#weather data
weather_cols = ["T_out", "Tdewpoint","RH_out","Press_mm_hg","Windspeed","Visibility"]

target = ["Appliances"]


# In[5]:


output = train_df[target]
input_vars = train_df[temp_cols + rho_cols + weather_cols]


# In[6]:


input_vars.head()


# In[7]:


input_vars.describe()


# In[8]:


output.values


# In[9]:


plt.figure(figsize=(8,6))
plt.scatter(range(output.shape[0]), np.sort(output.values))
plt.xlabel('index',fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.title("Target Distribution", fontsize=14)
plt.show()


# In[10]:


plt.figure(figsize=(12,8))
sns.distplot(output.values, bins=100, kde=False)
plt.xlabel('Target',fontsize=12)
plt.title('Target Histogram', fontsize=14)
plt.show()


# In[11]:


missing_df = input_vars.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df


# In[12]:


dtype_df = input_vars.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# In[13]:


unique_df = input_vars.nunique().reset_index()
unique_df.columns = ['col_name','unique_count']
constant_df = unique_df[unique_df['unique_count']==1]
constant_df.shape


# In[14]:


from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


# In[15]:


labels = []
values = []
for col in input_vars.columns:
    labels.append(col)
    values.append(spearmanr(input_vars[col].values, output.values)[0])


# In[16]:


corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')


# In[17]:


corr_df = corr_df[(corr_df['corr_values']>0.1) | (corr_df['corr_values']<-0.1)]
ind = np.arange(corr_df.shape[0])
width = 0.9
fig,ax = plt.subplots(figsize=(12,30))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='b')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


# In[18]:


cols_to_use = corr_df[(corr_df['corr_values']>0.11) | (corr_df['corr_values']<-0.11)].col_labels.tolist()

temp_df = train_df[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f,ax = plt.subplots(figsize=(20,20))

sns.heatmap(corrmat, vmax=1., square=True, cmap = 'YlGnBu', annot=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# In[19]:


train_X  = input_vars.drop(constant_df.col_name.tolist(), axis=1)
#test_X = test_df.drop(constant_df.col_name.tolist() + ["ID"], axis=1)
train_y = output.values


# In[20]:


from sklearn import ensemble
model = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5, n_jobs=-1, random_state=0)
model.fit(train_X, train_y)


# In[21]:


feature_names = train_X.columns.values
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]


# In[22]:


plt.figure(figsize=(12,12))
plt.title("Feature Importances")
plt.bar(range(len(indices)), importances[indices], color='r', yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feature_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


# In[23]:


from sklearn.linear_model import LinearRegression


# In[24]:


X_train = input_vars
y_train = output


# In[25]:


benchmark_model = LinearRegression()
benchmark_model.fit(X_train,y_train)


# In[26]:


test_data = pd.read_csv('test.csv')


# In[27]:


X_test = test_data[temp_cols + rho_cols + weather_cols]
y_test = test_data[target]


# In[28]:


benchmark_model.score(X_train,y_train)


# In[29]:


benchmark_model.score(X_test,y_test)


# <h1>Data Preprocessing</h1>

# In[45]:


train = X_train.drop(['T6', 'T9'], axis=1)
test  = X_test.drop(['T6', 'T9'], axis=1)
train = train.join(y_train)
test = test.join(y_test)


# In[46]:


from sklearn.preprocessing import StandardScaler


# In[47]:


standard_scaler = StandardScaler()

train_scaled = pd.DataFrame(columns=train.columns, index=train.index)
test_scaled = pd.DataFrame(columns=test.columns, index=test.index)

train_scaled[train_scaled.columns] = standard_scaler.fit_transform(train)
test_scaled[test_scaled.columns] = standard_scaler.fit_transform(test)


# In[50]:


X_train = train_scaled.drop("Appliances", axis=1)
y_train = train_scaled["Appliances"]

X_test = test_scaled.drop("Appliances", axis=1)
y_test = test_scaled["Appliances"]


# In[53]:


from sklearn.metrics import mean_squared_error

def pipeline(reg, X_train, y_train, X_test, y_test, **kwargs):
    reg_props = {}
    
    regressor = reg(**kwargs)
    regressor.fit(X_train, y_train)
    
    reg_props["name"] = reg.__name__
    reg_props["train_score"] = regressor.score(X_train, y_train)
    reg_props["test_score"] = regressor.score(X_test, y_test)
    reg_props["rmse"] = np.sqrt(mean_squared_error(y_test, regressor.predict(X_test)))
    
    return reg_props


# In[55]:


from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor


# In[59]:


seed=79
np.random.seed(seed)


# In[60]:


def execute_pipeline():
    regressors = [
        Ridge, 
        Lasso,
        RandomForestRegressor,
        ExtraTreesRegressor,
        MLPRegressor
    ]
    
    props = []
    
    for reg in regressors:
        properties = pipeline( reg, X_train, y_train, X_test, y_test, random_state=seed)
        props.append(properties)
        
    return props


# In[66]:


def get_properties():
    properties = execute_pipeline()
    
    names = [prop['name'] for prop in properties]
    train_scores = [prop['train_score'] for prop in properties]
    test_scores = [prop['test_score'] for prop in properties]
    rmse_vals = [prop['rmse'] for prop in properties]
    
    df = pd.DataFrame( index=names,
                      data={
                          "Training scores" : train_scores,
                          "Testing scores" : test_scores,
                          "RMSE" :rmse_vals
                      })
    return df


# In[67]:


properties = get_properties()


# In[68]:


benchmark_model.fit(X_train,y_train)


# In[73]:


properties = pd.concat([properties, pd.Series({
    "RMSE":np.sqrt(mean_squared_error(y_test, benchmark_model.predict(X_test))),
    "Training scores": benchmark_model.score(X_train,y_train),
    "Testing scores" : benchmark_model.score(X_test,y_test),
    "Name": "Linear Regression"
}).to_frame().T.set_index(["Name"])]
                      )


# In[74]:


properties


# In[75]:


ax= properties[["Training scores", "Testing scores", "RMSE"]].plot(kind="bar", title="Performance of each Regressor", figsize=(16, 8))
ax.set_ylabel("R2 Score/ RMSE", fontsize="large")


# Least  - LASSO<br>
# Best  - Extra Trees
