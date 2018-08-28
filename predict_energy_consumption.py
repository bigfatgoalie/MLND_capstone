
# coding: utf-8

# # Machine Learning Advanced Nanodegree
# ## Capstone Project
# ## Appliance Energy Prediction

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing, model_selection, metrics
#import lightgbm as lgb

get_ipython().run_line_magic('matplotlib', 'inline')

seed=79

from scipy.stats import spearmanr


# ## Reading data

# In[2]:


data = pd.read_csv('energydata_complete.csv')
data.describe()


# In[3]:


data.head()


# ## Exploratory Analysis

# In[4]:


print("Number of rows = {}".format(data.shape[0]))
print("Number of columns = {}".format(data.columns.shape[0]))


# In[5]:


print("Column wise count of null values:-")
print(data.isnull().sum())


# ### So there are no null values in any of the columns. Now, dividing the columns according to the type of data:

# In[6]:


#temperature columns
temp_cols = ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]

#humidity sensor columns
rho_cols = ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_6","RH_7","RH_8","RH_9"]

#weather data
weather_cols = ["T_out", "Tdewpoint","RH_out","Press_mm_hg",
                "Windspeed","Visibility"] 

randoms = ["rv1", "rv2"]

target = ["Appliances"]


# ### the variables 'Date', 'lights' don't matter 
# ### The problem is regression not time series so Date doesn't matter
# ### lights doesn't matter as we have to predict total energy not category wise energy

# In[7]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data,test_size=0.25,random_state=seed)


# In[8]:


train.describe()


# In[9]:


test.describe()


# In[10]:


input_vars = train[temp_cols+rho_cols+weather_cols+randoms]
output_var = train[target]


# In[11]:


input_vars.describe()


# In[12]:


output_var.describe()


# ### Observations
# 
# * Temperature ranges for all home sensors is between 14.89℃ to 29.86℃ except for T6 for which it is -6.06℃ to 28.29℃. The reason for such low readings is that the sensor is kept outside.
# * Similarly, humudity ranges for all home sensors is between 20.60`%` to 63.36%. Except for RH_5 and RH_6, whose ranges are 29.82`%` to 96.32`%` and 1`%` to 99.9`%` respectively.
#     * The reason behind this is that RH_5 is inside the bathroom,
#     * And RH_6 is outside the building, explaining the high humidity values.
# * One interesting observation can be seen in `Appliances` column that although the max consumption is 1080`Wh`, 75`%` of values are less than 100`Wh`. This shows that there are fewer cases when Appliance energy consumption is very high.

# ### VISUAL INPUT VARS ANALYSIS

# In[13]:


hists = input_vars.hist(figsize=(16, 16), bins=20,edgecolor='black')


# In[14]:


ax = output_var.hist(figsize=(4,4), bins=20,edgecolor='black')
for axe in ax.flatten():
    axe.set_xlabel("Energy Consumption(Wh)")
    axe.set_ylabel("Frequency")


# It can be observed from Histograms that:-
# * All humidity values except `RH_6` and `RH_out` follow a Normal distribution. That is, all the readings from sensors inside the home are from a Normal distribution.
# * Similarly, all temperature readings follow a Normal distribution except for `T9`.
# * Out of the remaining columns, we can see that `Visibility`, `Windspeed` and **`Appliances`** are skewed.
# * The random variables rv1 and rv2 have more or less the same values for all the recordings.
# 
# The output variable Appliances has most values less than 200Wh, showing that high energy consumption cases are very low.

# ### Now that we know the ranges and distribution of the attributes we need to check for inter-attribute correlation.

# ## Correlation plots

# In[15]:


from scipy.stats import spearmanr


# In[16]:


labels = []
values = []
for col in input_vars.columns:
    labels.append(col)
    values.append(spearmanr(input_vars[col].values, output_var.values)[0])


# In[17]:


corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')


# In[18]:


corr_df = corr_df[(corr_df['corr_values']>=0.1) | (corr_df['corr_values']<=-0.1)]


# In[19]:


rel_cols = corr_df.col_labels.tolist()


# In[20]:


temp_df = train[rel_cols]


# In[21]:


corrmat = temp_df.corr(method='spearman')
f,ax = plt.subplots(figsize=(20,20))

sns.heatmap(corrmat, vmax=1., square=True, cmap = 'YlGnBu', annot=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# In[22]:


input_vars.columns


# In[23]:


rel_cols


# ### This tells us all the relevant columns which have the most impact on the output variable (absolute correlation value >= 0.1)
# * The random variables rv1 and rv2 are not important
# * All the temperature variables from T1-T9 and T_out have high correlation with the target Appliances
# * Most of the humidity columns have been left out 
# * columns of Visibility, Tdewpoint, Press_mm_hg also have low correlation values 

# * The inter-attribute correlation is very high(>0.9) b/w T6 and T_out
# * A number of variables have high correlation with T9 (T3,T5,T7,T8), so it is clear that T9 is redundant
# ### T6 and T9 need to be removed 
# ### Although I've got the important variables based on correlation, its better to get a second opinion using a Tree based regressor (Extra Randomized Trees in this case).

# In[24]:


train_X = train[input_vars.columns]
train_Y = train[output_var.columns]


# In[25]:


train_X = train_X.drop(["T6", "T9"], axis=1)


# In[26]:


from sklearn import ensemble
model = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5, n_jobs=-1, random_state=0)
model.fit(train_X, train_Y)


# In[27]:


feature_names = train_X.columns.values
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]


# In[28]:


plt.figure(figsize=(16,16))
plt.title("Feature Importances")
plt.bar(range(len(indices)), importances[indices], color='b', align="center",edgecolor='red')
plt.xticks(range(len(indices)), feature_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


# In[29]:


feature_names[indices]


# In[30]:


rel_cols


# * Comparing both lists we see that rv1, rv2 , Visibility are the least important of the lot. So removing them and keeping the rest.

# In[31]:


train_X = train_X.drop(["rv1","rv2","Visibility"],axis=1)


# In[32]:


test_X = test[input_vars.columns]
test_Y = test[output_var.columns]


# In[33]:


test_X.head()


# In[34]:


test_X.drop(["T6", "T9","rv1","rv2","Visibility"], axis=1, inplace=True)


# In[35]:


test_X.head()


# In[36]:


train_X.head()


# In[37]:


test_X.columns


# In[38]:


train_X.columns


# In[39]:


# Import scaler
from sklearn.preprocessing import StandardScaler

# Scales the data to zero mean and unit variance
standard_scaler = StandardScaler()


# In[40]:


train = train[list(train_X.columns.values) + target]


# In[41]:


test = test[list(test_X.columns.values) + target]


# In[42]:


train.head()


# In[43]:


# Create dummy dataframes to hold the scaled train and test data
train_scaled = pd.DataFrame(columns=train.columns, index=train.index)
test_scaled = pd.DataFrame(columns=test.columns, index=test.index)


# In[44]:


train_scaled[train_scaled.columns] = standard_scaler.fit_transform(train)
test_scaled[test_scaled.columns] = standard_scaler.fit_transform(test)


# In[45]:


# Prepare training and testing data
train_X = train_scaled.drop("Appliances", axis=1)
train_Y = train_scaled["Appliances"]

test_X = test_scaled.drop("Appliances", axis=1)
test_Y = test_scaled["Appliances"]


# In[46]:


from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor


# In[47]:


regressors = [
        Ridge, 
        Lasso,
        RandomForestRegressor,
        GradientBoostingRegressor,
        ExtraTreesRegressor,
        MLPRegressor
    ]


# In[48]:


props = []


# In[49]:


for reg in regressors:
        regs = reg(random_state=seed)
        regs.fit(train_X, train_Y)
        reg_props = {}
        reg_props["name"] = reg.__name__
        reg_props["train_score"] = regs.score(train_X, train_Y)
        reg_props["test_score"] = regs.score(test_X, test_Y)
        props.append(reg_props)


# In[50]:


props


# In[51]:


names = [prop["name"] for prop in props]
train_scores = [prop["train_score"] for prop in props]
test_scores = [prop["test_score"] for prop in props]

df = pd.DataFrame(index=names, 
                  data = {
                            "Training scores": train_scores,
                            "Testing scores": test_scores
                         }
                  )


# In[52]:


df


# * Least performing Regressor - Lasso Regressor
# * Best performing Regressor - Extra Trees Regressor
# 
# 
# Even though Extra Trees Regressor has a R2 score of 1.0 on traininig set, which might suggest overfitting but, it has the highest score on test set. Clearly, ExtraTreesRegressor is the best model out of given models.

# ## Hyperparameter Tuning

# In[53]:


from sklearn.model_selection import GridSearchCV

# Initialize the best performing regressor
clf = ExtraTreesRegressor(random_state=seed)

# Define the parameter subset
param_grid = {
    "n_estimators": [10, 50, 100, 200, 250],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [None, 10, 50, 100, 200, 500]
}

# Use Randomized search to try 20 subsets from parameter space with 5-fold cross validation
grid_search = GridSearchCV(clf, param_grid, scoring="r2", cv=5, n_jobs=-1, verbose=2)
grid_search.fit(train_X, train_Y)


# In[54]:


grid_search.best_params_


# In[55]:


grid_search.best_estimator_


# In[56]:


grid_search.best_estimator_.score(train_X,train_Y)


# In[57]:


grid_search.best_estimator_.score(test_X,test_Y)


# In[58]:


feature_indices = np.argsort(grid_search.best_estimator_.feature_importances_)


# In[59]:


print("Top 5 most important features:-")
# Reverse the array to get important features at the beginning
for index in feature_indices[::-1][:5]:
    print(train_X.columns[index])
    
print("\nTop 5 least important features:-")
for index in feature_indices[:5]:
    print(test_X.columns[index])


# In[60]:


feature_names = train_X.columns.values
importances = grid_search.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]


# In[61]:


plt.figure(figsize=(16,16))
plt.title("Feature Importances")
plt.bar(range(len(indices)), importances[indices], color='b', align="center")
plt.xticks(range(len(indices)), feature_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


# In[62]:


# Constructing data set from reduced feature space
train_X_reduced = train_X[train_X.columns[feature_indices[::-1][:5]]]
test_X_reduced = test_X[test_X.columns[feature_indices[::-1][:5]]]


# In[63]:


from sklearn.base import clone

# Clone the best model
reg_best = clone(grid_search.best_estimator_)
# Fit the model on reduced data set 
reg_best.fit(train_X_reduced, train_Y)


# In[64]:


reg_best.score(test_X_reduced, test_Y)


# In[65]:


#Difference is about 10.5%
from sklearn.model_selection import cross_validate


# In[66]:


clf = grid_search.best_estimator_


# In[67]:


data_X = train_X.append(test_X)
data_Y = train_Y.append(test_Y)


# In[68]:


scores = cross_validate(clf,data_X,data_Y, cv=5)


# In[69]:


scores.keys()


# In[70]:


scores['test_score']


# In[71]:


scores['train_score']

