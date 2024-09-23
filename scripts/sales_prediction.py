import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

class SalesPrediction:
    def __init__(self, train_path, test_path):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.pipeline = None
        self.rf_model = None
        self.lstm_model = None
        self.scaler = None

    def feature_engineering(self):
        for df in [self.train_df, self.test_df]:
            df['Date'] = pd.to_datetime(df['Date'])
            df['DayOfWeek'] = df['Date'].dt.dayofweek + 1

            holidays = df[df['StateHoliday'] != '0']['Date'].unique()
            df['DaysToHoliday'] = df['Date'].apply(lambda x: min((holidays - x).days) if holidays.size > 0 else np.nan)
            df['DaysAfterHoliday'] = df['Date'].apply(lambda x: min((x - holidays).days) if holidays.size > 0 else np.nan)

            df['IsBeginningOfMonth'] = (df['Date'].dt.day <= 10).astype(int)
            df['IsMidMonth'] = ((df['Date'].dt.day > 10) & (df['Date'].dt.day <= 20)).astype(int)
            df['IsEndOfMonth'] = (df['Date'].dt.day > 20).astype(int)

            df['Quarter'] = df['Date'].dt.quarter
            df['WeekOfYear'] = df['Date'].dt.isocalendar().week

    def clean_competition_distance(self, x):
        if isinstance(x, str):
            if x == '1-5km':
                return 3000  # Midpoint of 1-5km in meters
            elif '-' in x:
                lower, upper = map(lambda y: float(y.replace('km', '')) * 1000, x.split('-'))
                return (lower + upper) / 2
            else:
                return float(x.replace('km', '')) * 1000
        return x
    
    def pre_process(self):
        # Check if the 'Date' column is in the DataFrame
        if 'Date' not in self.train_df.columns or self.train_df['Date'].isnull().all():
            raise ValueError("The 'Date' column is missing or has no valid entries.")

        # Clean the 'CompetitionDistance' column
        self.train_df['CompetitionDistance'] = self.train_df['CompetitionDistance'].apply(self.clean_competition_distance)
        self.test_df['CompetitionDistance'] = self.test_df['CompetitionDistance'].apply(self.clean_competition_distance)

        # Handle categorical columns
        categorical_cols = ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval', 'CompetitionDistanceBin']
        self.train_df = pd.get_dummies(self.train_df, columns=categorical_cols, drop_first=True)
        self.test_df = pd.get_dummies(self.test_df, columns=categorical_cols, drop_first=True)

        # Ensure the columns in the test set match the train set after one-hot encoding
        self.test_df = self.test_df.reindex(columns=self.train_df.columns.drop('Sales'), fill_value=0)

        # Handle datetime columns
        self.train_df['Date'] = pd.to_datetime(self.train_df['Date'], errors='coerce')
        self.test_df['Date'] = pd.to_datetime(self.test_df['Date'], errors='coerce')

        if self.train_df['SalesPerCustomer'].isnull().any():
            # Fill missing values in 'SalesPerCustomer' with the mean value
            self.train_df['SalesPerCustomer'].fillna(self.train_df['SalesPerCustomer'].mean(), inplace=True)

        # Numeric column scaling
        numeric_cols = self.train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Sales' in numeric_cols:
            numeric_cols.remove('Sales')

        self.scaler = StandardScaler()
        self.train_df[numeric_cols] = self.scaler.fit_transform(self.train_df[numeric_cols])
        self.test_df[numeric_cols] = self.scaler.transform(self.test_df[numeric_cols])

        # Define feature columns for training
        self.feature_cols = [col for col in self.train_df.columns if col != 'Sales']

    def model_training(self):
        X = self.train_df[self.feature_cols]
        y = self.train_df['Sales']

        # drop the Date column
        X = X.drop('Date', axis = 1)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)

        # Validate the model
        val_predictions = self.rf_model.predict(X_val)
        mse = mean_squared_error(y_val, val_predictions)
        mae = mean_absolute_error(y_val, val_predictions)
        print(f"Validation MSE: {mse}")
        print(f"Validation MAE: {mae}")

    def deep_learning_model(self):
        # Filter the data for Store 1
        store_data = self.train_df[self.train_df['Store'] == 1][['Date', 'Sales']]
        
        # Convert 'Date' to datetime format if not already done
        store_data['Date'] = pd.to_datetime(store_data['Date'], errors='coerce')
        
        # Set the 'Date' column as the index
        store_data.set_index('Date', inplace=True)

        # Resample the data by day ('D'), summing up sales for each day
        time_series_data = store_data.resample('D').sum()

        # Perform ADF test (Augmented Dickey-Fuller) to check for stationarity
        result = adfuller(time_series_data['Sales'].dropna())
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])

        # If the series is not stationary (p-value > 0.05), difference it
        if result[1] > 0.05:
            time_series_data['Sales'] = time_series_data['Sales'].diff().dropna()

        # Plot Autocorrelation Function (ACF)
        plot_acf(time_series_data['Sales'].dropna())
        plt.title('Autocorrelation')
        plt.show()

        # Plot Partial Autocorrelation Function (PACF)
        plot_pacf(time_series_data['Sales'].dropna())
        plt.title('Partial Autocorrelation')
        plt.show()

        return time_series_data