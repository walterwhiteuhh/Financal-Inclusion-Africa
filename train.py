# dataframe and plotting
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
#from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
#from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# function to preprocess our data from train models

enc = LabelEncoder()

def preprocessing_data(df):
    # convert categorical features to numerical features
    # # checking categorical columns
    categorical_columns = df.select_dtypes(include = 'object')
    categorical_columns

    # categorical features to be converted by One Hot Encoding
    categ =  ["relationship_with_head", "marital_status", "education_level", "job_type", "country"]
    # One Hot Encoding conversion
    df = pd.get_dummies(df, prefix_sep='_', columns = categ)

    # Labelncoder conversion
    # #for i in categ:
    # df[i] = enc.fit_transform(df[i])
    df['location_type'] = enc.fit_transform(df['location_type'])
    df['cellphone_access'] = enc.fit_transform(df['cellphone_access'])
    df['gender_of_respondent'] = enc.fit_transform(df['gender_of_respondent'])    
    
    # drop uniquid column
    df = df.drop(["uniqueid"], axis=1)
    
    # scale our data into range of 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)
    
    return df                  

# später auch test_end Set!


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

# Explore Target distribution 
#sns.catplot(x="country", y="bank_account", kind="count",  data=train)

## number of people with bank account per country
f,axes=plt.subplots(1,3,figsize=(16,5))
sns.countplot(x='bank_account',hue='country',data=train,palette='magma',ax=axes[0]).set_title('People with bank account per country',fontsize=(plotfontsize))
sns.countplot(x='bank_account',data=train,ax=axes[1]).set_title('People with bank account',fontsize=(plotfontsize))
sns.countplot(x='bank_account',hue='location_type', data=train,palette='magma', ax=axes[2]).set_title('People without [0] vs. with [1] bank account per location type',fontsize=(plotfontsize))
plt.show()

## finding correlation 
data_corr=train.copy()
data_corr.drop('uniqueid',inplace=True,axis=1)

ll=LabelEncoder()
data_corr['country']=ll.fit_transform(data_corr['country'])
data_corr['bank_account']=ll.fit_transform(data_corr['bank_account'])
data_corr['location_type']=ll.fit_transform(data_corr['location_type'])
data_corr['cellphone_access']=ll.fit_transform(data_corr['cellphone_access'])
data_corr['gender_of_respondent']=ll.fit_transform(data_corr['gender_of_respondent'])
data_corr['relationship_with_head'] = ll.fit_transform(data_corr['relationship_with_head'])
data_corr['marital_status']=ll.fit_transform(data_corr['marital_status'])
data_corr['education_level']=ll.fit_transform(data_corr['education_level'])
#data_corr['job_type']=ll.fit_transform(data_corr['job_type'])
### correlation
sns.set_style('dark')
f,axes=plt.subplots(1,1,figsize=(15,6))
sns.heatmap(data_corr.corr(),vmin=0,vmax=1,annot=True)
plt.show()

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

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
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

# print confusion matrix baseline-model
cm = confusion_matrix(y_test, y_pred_baseline)
sns.heatmap(cm, cmap="YlGnBu", annot=True, fmt='d');
# print confusion matrix lg-model
cm = confusion_matrix(y_test, y_pred_lg)
sns.heatmap(cm, cmap="YlGnBu", annot=True, fmt='d');

# Print classification report for more information
print(classification_report(y_test, y_pred_baseline))
print(classification_report(y_test, y_pred_lg))

plt.show()