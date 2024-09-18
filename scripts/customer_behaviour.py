import pandas as pd
import numpy as np
from IPython.display import display

class CustomerBehaviour():
    def __init__(self, store_path, train_path, test_path):
        self.store_df = pd.read_csv(store_path)
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        display("Data loaded successfully")

    def merge_data(self):
        self.train_merged = pd.merge(self.train_df, self.store_df, on = 'Store')
        self.test_merged = pd.merge(self.test_df, self.store_df, on = 'Store')
        display("Data merged successfully")
    
    def clean_data(self):
        for df in [self.train_merged, self.test_merged]:
            display(df.head())

            # Calculate missing values
            missing_values = df.isnull().sum()
            total_values = len(df)

            # Calculate missing value percentage
            missing_percentage = (missing_values / total_values) * 100

            # Concatenate the two Series
            combined_missing = pd.concat([missing_values, missing_percentage], axis=1)
            combined_missing.columns = ['Missing Values', 'Percentage(%)']
            display(combined_missing)

            # Handle outliers using z-score (only for numeric columns)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[f'{col}_z'] = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[df[f'{col}_z'] > 3]
                display(f"Number of outliers in {col}: {len(outliers)}")

                # Replace outliers with median
                median_value = df[col].median()
                df.loc[df[f'{col}_z'] > 3, col] = median_value

                df.drop(columns=f'{col}_z', inplace=True)

            # Convert date to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df['Month'] = df['Date'].dt.month
                df['Year'] = df['Date'].dt.year
                df['DayOfWeek'] = df['Date'].dt.dayofweek

        display("Data cleaned successfully.")