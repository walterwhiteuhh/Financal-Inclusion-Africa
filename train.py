import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load files into a pandas dataframe
train = pd.read_csv('data/Train.csv')

# Let’s observe the shape of our datasets.
print('train data shape :', train.shape)

# inspect train data
train.head()
train.info()
train.describe()
# Check for missing values
print('missing values:', train.isnull())
print('missing values:', train.isnull().sum())

plotfontsize = 10

# number of people with bank account and per country and location type
f,axes=plt.subplots(1,3,figsize=(16,5))
sns.countplot(x='bank_account',hue='country',data=train,palette='magma',ax=axes[0]).set_title('Bank account per country',fontsize=(plotfontsize))
sns.countplot(x='bank_account',data=train,ax=axes[1]).set_title('People with and without bank account',fontsize=(plotfontsize))
sns.countplot(x='bank_account',hue='location_type', data=train,palette='magma', ax=axes[2]).set_title('Bank account per location type',fontsize=(plotfontsize))
plt.savefig("bank_accounts_per_country_and_location_type.pdf")

# Drop uniqueid column 
data_corr=train.copy()
data_corr.drop('uniqueid',inplace=True,axis=1)

# Transform categorical features to numerical values
ll=LabelEncoder()
data_corr['country']=ll.fit_transform(data_corr['country'])
data_corr['bank_account']=ll.fit_transform(data_corr['bank_account'])
data_corr['location_type']=ll.fit_transform(data_corr['location_type'])
data_corr['cellphone_access']=ll.fit_transform(data_corr['cellphone_access'])
data_corr['gender_of_respondent']=ll.fit_transform(data_corr['gender_of_respondent'])
data_corr['relationship_with_head'] = ll.fit_transform(data_corr['relationship_with_head'])
data_corr['marital_status']=ll.fit_transform(data_corr['marital_status'])
data_corr['education_level']=ll.fit_transform(data_corr['education_level'])
data_corr['job_type']=ll.fit_transform(data_corr['job_type'])

# Plot correlation of all features
sns.set_style('dark')
f,axes=plt.subplots(1,1,figsize=(15,6))
sns.heatmap(data_corr.corr(),vmin=0,vmax=1,annot=True)
plt.savefig("heatmap_correlations.pdf")

y = train.bank_account  # Define target
X = train.drop('bank_account', axis=1)  # Define predictors 

# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get a data set with X and y test data
train_visual = pd.concat([X_train,y_train],axis=1)

categorical_columns = X_train.select_dtypes(include = 'object')
categorical_columns = X_test.select_dtypes(include = 'object')
#categorical_columns = test_end.select_dtypes(include = 'object')

X_train.columns

X_train.head()

for i in categorical_columns:
    enc = LabelEncoder()
    print(i)
    X_train[i] = enc.fit_transform(X_train[i])
    X_test[i] = enc.fit_transform(X_test[i])
    y_train = enc.fit_transform(y_train)
    y_test = enc.fit_transform(y_test)

train_visual.head()

# checking categorical columns
categorical_columns = train.select_dtypes(include = 'object')
categorical_columns

# Defining baseline model location_type / Urban = 1 Rural = 0

def baseline_model(X_train):
    y_pred_baseline = [1 if x == 1 else 0 for x in X_train.location_type]
    return y_pred_baseline

# create Logistic Regression Model
lg_model = LogisticRegression()

# Instantiate model and fit it on train data
lg_model.fit(X_train, y_train)

# Make predictions for test set
y_pred_baseline = baseline_model(X_test)
y_pred_lg = lg_model.predict(X_test)

# print confusion matrix of baseline-model
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
sns.heatmap(cm_baseline, cmap="YlGnBu", annot=True, fmt='d');
plt.savefig("confusion_matrix_baseline_model.pdf")

# print confusion matrix lg-model
cm_lg_model = confusion_matrix(y_test, y_pred_lg)
sns.heatmap(cm_lg_model, cmap="YlGnBu", annot=True, fmt='d');
plt.savefig("confusion_matrix_lg_model.pdf")

# Print classification report for more information of the lg_model
print(classification_report(y_test, y_pred_baseline))
print(classification_report(y_test, y_pred_lg))

# my model 1: RandomForestClassifier
my_model_1 = RandomForestClassifier(n_estimators=100, random_state= 20, n_jobs = 2)

# Instantiate model and fit it on train data
my_model_1.fit(X_train, y_train)

# Make predictions for test set
y_pred_my_model_1 = my_model_1.predict(X_test)

# print confusion matrix of Model 1: Random Forest on train data
cm_my_model_1 = confusion_matrix(y_test, y_pred_my_model_1)
sns.heatmap(cm_my_model_1, cmap="YlGnBu", annot=True, fmt='d');
plt.savefig("confusion_matrix_model_1.pdf")

# Print classification report for my_model_1 on test data
print(classification_report(y_test, y_pred_my_model_1))

# My Model 2: Optimize model paramaters via Grid Search
param_grid = {#'class_weight': [2, 5, 10],
           # 'gamma': [0.5, 1, 1.5, 2, 5],
            'max_samples': [1, 10, 100],
           # 'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [5, 10, 100],
            'verbose': [1, 5, 10]
            }

my_model_2 = GridSearchCV(my_model_1, param_grid)

# Instantiate model and fit it on train data
my_model_2.fit(X_train, y_train)
print(my_model_2.best_params_)

# Make predictions for test set
y_pred_my_model_2 = my_model_2.predict(X_test)

# print confusion matrix of Model 2 as Optimized Model 1 with Grid Search
cm_my_model_2 = confusion_matrix(y_test, y_pred_my_model_2)
sns.heatmap(cm_my_model_2, cmap="YlGnBu", annot=True, fmt='d');
plt.savefig("confusion_matrix_model_2.pdf")

# Print classification report for my_model_2 on test data
print(classification_report(y_test, y_pred_my_model_2))

# Print again classification report for my_model_1 on test data
print(classification_report(y_test, y_pred_my_model_1))

# Print again classification report for lg_model
print(classification_report(y_test, y_pred_lg))

# Print precision for the three models
print("Precision of LogisticRegression: {}".format(precision_score(y_test, y_pred_lg)))
print("Precision of Model 1: RandomForests: {}".format(precision_score(y_test, y_pred_my_model_1)))
print("Precision of Model 2: Optimized Model 1: {}".format(precision_score(y_test, y_pred_my_model_2)))
# classifiers = {
#     "Baseline Model": baseline_model(),
#     "Logistic Regression": lg_model(),
#     "Model 1 = RandomForestClassifier": my_model_1(),
#     "Model 2 = GridSearchCV of Model 1": my_model_2(),
# }

# f, axes = plt.subplots(1, 4, figsize=(20, 5), sharey='row')