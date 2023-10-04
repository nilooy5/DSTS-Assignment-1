from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Part B: Predictive Modelling
# I. Feature engineering
df = pd.read_csv('data/zomato_df_final_data.csv')

df_rating_text = pd.get_dummies(df['rating_text'])
df = df.drop(
    columns=["address", "link", "phone", "title", "color", "cuisine_color", "type", "cuisine", "rating_text", "lat",
             "lng"], axis=1)

df_rating_text['bin_rating'] = 0
# if in df_rating_text['Poor'] == True or df_rating_text['Average'] == True then put 1 in df_rating_text[
# 'bin_rating'] else put 2
df_rating_text['bin_rating'] = df_rating_text.apply(lambda x: 1 if x['Poor'] == 1 or x['Average'] == 1 else 2, axis=1)

df.groupon = df.groupon.astype(int)

# cleaning up the subzone column
df['subzone'] = df['subzone'].apply(lambda x: x.split(',')[-1].strip() if ',' in x else x)

# encoding the subzone column
label_encoder = LabelEncoder()
df['subzone_encoded'] = label_encoder.fit_transform(df['subzone'])

# handling the missing values
# fill the rating column with the mean
df['rating_number'] = df.groupby('subzone_encoded')['rating_number'].transform(lambda x: x.fillna(x.mean()))
df['rating_number'] = df['rating_number'].fillna(df['rating_number'].mean())
# fill the votes column with the mean
df['votes'] = df.groupby('subzone_encoded')['votes'].transform(lambda x: x.fillna(x.mean()))
df['votes'] = df['votes'].fillna(df['votes'].mean())

df['cost'] = df.groupby('subzone_encoded')['cost'].transform(lambda x: x.fillna(x.mean()))
df['cost'] = df['cost'].fillna(df['cost'].mean())

df['cost_2'] = df.groupby('subzone_encoded')['cost_2'].transform(lambda x: x.fillna(x.mean()))
df['cost_2'] = df['cost_2'].fillna(df['cost_2'].mean())

df = df.drop(columns=["subzone"], axis=1)

# II. Regression model
X = df.drop(columns=['rating_number'])
y = df['rating_number']

# Build a linear regression model (model_regression_1) to predict the restaurants rating (numeric rating) from other
# features (columns) in the dataset. Please consider splitting the data into train (80%) and test (20%) sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
model_regression_1 = LinearRegression()
model_regression_1.fit(X_train, y_train)
# predictions
y_pred = model_regression_1.predict(X_test)
# print('Coefficients: \n', model_regression_1.coef_)
# Evaluate the model using the Mean Squared Error and R-squared metrics.
print('Regression Model 1:')
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
print('')
print('')
print('')

# Build another linear regression model (model_regression_2) with using the Gradient Descent as the optimisation
# function

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_regression_2 = SGDRegressor(loss="squared_error", max_iter=150, random_state=0)
model_regression_2.fit(X_train_scaled, y_train)
# predictions
y_pred = model_regression_2.predict(X_test_scaled)
# print('Coefficients: \n', model_regression_2.coef_)
# Evaluate the model using the Mean Squared Error and R-squared metrics.
print('Regression Model 2:')
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
print('')
print('')
print('')


df['bin_rating'] = df_rating_text['bin_rating']


X = df.drop(columns=['bin_rating'])
y = df['bin_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
model_classification_3 = LogisticRegression()
model_classification_3.fit(X_train, y_train)
# predictions
y_pred = model_classification_3.predict(X_test)
# print('Coefficients: \n', model_classification_3.coef_)
# Evaluate the model using the Accuracy, Confusion Matrix and Classification Report metrics.
print('Classification Model 3:')
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
print('Classification Report: \n', classification_report(y_test, y_pred))
