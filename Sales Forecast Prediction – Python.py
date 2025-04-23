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
plt.title('Sales trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.legend()
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#4. Feature Engineering – Creating Lagged Features

def create_lagged_features(data, lag=1):
    lagged_data = data.copy()
    for i in range(1, lag+1):
        lagged_data[f'lag_{i}'] = lagged_data['Sales'].shift(i)
    return lagged_data
lag = 5
sales_with_lags = create_lagged_features(data[['Order Date', 'Sales']], lag)
sales_with_lags = sales_with_lags.dropna()

#5. Preparing the Data for Training

X = sales_with_lags.drop(columns=['Order Date', 'Sales'])
y = sales_with_lags['Sales']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)

#Training the XGBoost Model

model