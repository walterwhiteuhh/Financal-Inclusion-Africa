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
from sklearn.metrics import confusion_matrix, classification_report
#from google.colab import files
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

# Explore Target distribution 
#sns.catplot(x="country", y="bank_account", kind="count",  data=train)

## number of people with bank account per country
f,axes=plt.subplots(1,3,figsize=(16,5))
sns.countplot(x='bank_account',hue='country',data=train,palette='magma',ax=axes[0]).set_title('People with bank account per country',fontsize=(13))
sns.countplot(x='bank_account',data=train,ax=axes[1]).set_title('People with bank account',fontsize=(13))
sns.countplot(x='bank_account',hue='location_type', data=train,palette='magma', ax=axes[2]).set_title('People without [0] vs. with [1] bank account per location type',fontsize=(13))
plt.show()

# Define predictors and target
y = train.bank_account
X = train.drop('bank_account', axis=1)

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

train_visual.head()

# checking categorical columns
categorical_columns = train.select_dtypes(include = 'object')
categorical_columns

# create Logistic Regression Model
lg_model = LogisticRegression()

# Instantiate model and fit it on train data
lg_model.fit(X_train, y_train)

# Make predictions for test set
y_pred_lg = lg_model.predict(X_test)
# print confusion matrix
cm = confusion_matrix(y_test, y_pred_lg)
sns.heatmap(cm, cmap="YlGnBu", annot=True, fmt='d');

# Print classification report for more information
print(classification_report(y_test, y_pred_lg))