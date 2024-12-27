#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance


# In[2]:


import yfinance as yf


# In[3]:


gold = yf.download('GC=F', period='10y')
gold


# In[4]:


gold.reset_index(inplace=True)
gold


# In[5]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# In[6]:


gold.isnull().sum()


# In[7]:


gold.describe()


# In[8]:


gold.shape


# In[9]:


gold.info()


# In[10]:


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Download historical gold data for the last 10 years
gold = yf.download('GC=F', period='10y')

# Reset the index to have 'Date' as a column
gold.reset_index(inplace=True)

# Ensure the 'Date' column is in datetime format
gold['Date'] = pd.to_datetime(gold['Date'])

# Resample the data to 6-month intervals using the 'Date' column
gold_6months = gold.resample('6M', on='Date').mean()

# Plot the closing price trend over the 6-month intervals using the index
plt.figure(figsize=(10, 6))
plt.plot(gold_6months.index, gold_6months['Close'], marker='o', color='b', linestyle='-', label='Gold Price (6-month intervals)')
plt.title('Gold Price Trend Over Time (6-Month Intervals)')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()


# In[11]:


correlation = gold.corr()


# In[12]:


plt.figure(figsize=(6,6))

sns.heatmap(correlation,cbar=True, square=True, annot=True, annot_kws={"size":8}, cmap="Blues")


# In[13]:


print(correlation['Close'])


# In[14]:


sns.displot(gold['Close'], color='green')


# In[15]:


X = gold.drop(['Date','Close'], axis=1)
y = gold['Close']


# In[16]:


y.shape


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


reg = RandomForestRegressor()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
r2_score(y_test, pred)


# In[19]:


from sklearn.linear_model import Ridge
reg = Ridge(alpha=0.5)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
r2_score(y_test, pred)


# In[20]:


from sklearn.linear_model import Lasso
reg = Lasso(alpha=0.5)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
r2_score(y_test, pred)


# In[21]:


print(pred)


# In[ ]:





# In[22]:


# Retrieve the corresponding dates from the original dataframe for the test set
test_dates = gold.loc[X_test.index, 'Date']

# Create a DataFrame to display dates, actual values, and predicted values
output_df = pd.DataFrame({
    'Date': test_dates,
    'Actual Close': y_test,
    'Predicted Close': pred
})

# Print the final output
print(output_df)


# In[29]:


output_df.to_csv('output.csv')


# In[25]:


# Download historical gold data for the last 10 years
gold = yf.download('GC=F', period='10y')

# Reset index to include 'Date' as a column
gold.reset_index(inplace=True)

# Fill missing values if any (optional step)
gold.fillna(method='ffill', inplace=True)

# Define feature columns and target column
# Use all columns except 'Date' and 'Close' as features
X = gold.drop(['Date', 'Close'], axis=1)
y = gold['Close']  # Target variable: the closing price of gold

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

# Evaluate the model performance (optional)
pred = reg.predict(X_test)
score = r2_score(y_test, pred)
print(f"Model R2 Score: {score}")

# Get today's intraday data for prediction (1-hour interval)
today = yf.download('GC=F', period='1d', interval='1h')  # Download data for today
today.reset_index(inplace=True)  # Reset index to use 'Datetime' as a column

# Drop 'Datetime' and 'Close' columns and use the last available data for prediction
latest_features = today.drop(['Datetime', 'Close'], axis=1).iloc[-1].values.reshape(1, -1)

# Predict tomorrow's gold price
predicted_tomorrow = reg.predict(latest_features)

# Output the predicted price for tomorrow
print(f"Predicted Gold Price for Tomorrow: {predicted_tomorrow[0]}")


# In[26]:


import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Download historical gold data for the last 10 years
gold = yf.download('GC=F', period='10y')

# Reset index to include 'Date' as a column
gold.reset_index(inplace=True)

# Fill missing values if any (optional step)
gold.fillna(method='ffill', inplace=True)

# Define feature columns and target column
# Use all columns except 'Date' and 'Close' as features
X = gold.drop(['Date', 'Close'], axis=1)
y = gold['Close']  # Target variable: the closing price of gold

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

# Evaluate the model performance on training data
train_pred = reg.predict(X_train)
train_score = r2_score(y_train, train_pred)

# Evaluate the model performance on test data
test_pred = reg.predict(X_test)
test_score = r2_score(y_test, test_pred)

# Output R2 Scores
print(f"Training R2 Score: {train_score}")
print(f"Test R2 Score: {test_score}")

# Check for overfitting
if train_score - test_score > 0.1:  # Threshold can be adjusted
    print("Warning: The model may be overfitting.")
else:
    print("The model is performing consistently on both training and test data.")

# Get today's intraday data for prediction (1-hour interval)
today = yf.download('GC=F', period='1d', interval='1h')  # Download data for today
today.reset_index(inplace=True)  # Reset index to use 'Datetime' as a column

# Drop 'Datetime' and 'Close' columns and use the last available data for prediction
latest_features = today.drop(['Datetime', 'Close'], axis=1).iloc[-1].values.reshape(1, -1)

# Predict tomorrow's gold price
predicted_tomorrow = reg.predict(latest_features)

# Output the predicted price for tomorrow
print(f"Predicted Gold Price for Tomorrow: {predicted_tomorrow[0]}")


# In[27]:


import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Download historical gold data for the last 10 years
gold = yf.download('GC=F', period='10y')

# Reset index to include 'Date' as a column
gold.reset_index(inplace=True)

# Fill missing values if any (optional step)
gold.fillna(method='ffill', inplace=True)

# Define feature columns and target column
# Use all columns except 'Date' and 'Close' as features
X = gold.drop(['Date', 'Close'], axis=1)
y = gold['Close']  # Target variable: the closing price of gold

# Train a Random Forest model on the entire dataset
reg = RandomForestRegressor()
reg.fit(X, y)

# Evaluate the model performance on the entire dataset
full_dataset_pred = reg.predict(X)
full_dataset_score = r2_score(y, full_dataset_pred)

# Output R2 Score for the full dataset
print(f"R2 Score on Full Dataset: {full_dataset_score}")

# Split data into training and validation sets (20% held out for validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training set
reg.fit(X_train, y_train)

# Evaluate the model performance on the validation set
val_pred = reg.predict(X_val)
val_score = r2_score(y_val, val_pred)

# Output R2 Score for the validation set
print(f"R2 Score on Validation Set: {val_score}")

# Check for overfitting
if full_dataset_score - val_score > 0.1:  # Threshold can be adjusted
    print("Warning: The model may be overfitting.")
else:
    print("The model is performing consistently on both the full dataset and the validation set.")

# Get today's intraday data for prediction (1-hour interval)
today = yf.download('GC=F', period='1d', interval='1h')  # Download data for today
today.reset_index(inplace=True)  # Reset index to use 'Datetime' as a column

# Drop 'Datetime' and 'Close' columns and use the last available data for prediction
latest_features = today.drop(['Datetime', 'Close'], axis=1).iloc[-1].values.reshape(1, -1)

# Predict tomorrow's gold price
predicted_tomorrow = reg.predict(latest_features)

# Output the predicted price for tomorrow
print(f"Predicted Gold Price for Tomorrow: {predicted_tomorrow[0]}")


# In[28]:


import pandas as pd
import yfinance as yf

# Download historical gold data for the last 10 years
gold = yf.download('GC=F', period='10y')

# Reset index to include 'Date' as a column
gold.reset_index(inplace=True)

# Fill missing values if any (optional step)
gold.fillna(method='ffill', inplace=True)

# Display the shape and first few rows of the dataset
print("Dataset Shape:", gold.shape)
print("First few rows of the dataset:")
print(gold.head())

# Check for duplicate columns
duplicate_columns = gold.columns[gold.columns.duplicated()]
print("Duplicate Columns:", duplicate_columns)

# Select numeric columns only for correlation analysis
numeric_data = gold.select_dtypes(include=['float64', 'int64'])
print("Numeric Data Shape:", numeric_data.shape)
print("Unique Values in Each Numeric Column:")
print(numeric_data.nunique())

# Calculate and print the correlation matrix
correlation_matrix = numeric_data.corr()
print("Correlation Matrix:")
print(correlation_matrix)


# In[ ]:





# In[ ]:





# In[ ]:




