import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#loading the data set
file_path = 'train.csv'
data = pd.read_csv(file_path)
data.head()

# 3. Data Preprocessing and Visualization

data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y')

sales_by_date = data.groupby('Order Date')['Sales'].sum().reset_index()

plt.figure(figsize=(12,6))
plt.plot(sales_by_date['Order Date'], sales_by_date['Sales'], label= 'Sales', color='red')