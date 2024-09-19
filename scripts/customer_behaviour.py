import pandas as pd
import numpy as np
from IPython.display import display
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

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
        for df_name, df in [('Train Merged Data', self.train_merged), ('Test Merged Data', self.test_merged)]:
            display(f"---{df_name}---")
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

            # Handle missing values
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype == 'object':  
                        mode_value = df[col].mode()[0]
                        df[col].fillna(mode_value, inplace=True)
                    else:  
                        median_value = df[col].median()
                        df[col].fillna(median_value, inplace=True)
            if 'StateHoliday' in df.columns:
                df['StateHoliday'] = df['StateHoliday'].astype(str)

            # Convert date to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df['Month'] = df['Date'].dt.month
                df['Year'] = df['Date'].dt.year
                df['DayOfWeek'] = df['Date'].dt.dayofweek

            # Check for any remaining missing values
            remaining_missing = df.isnull().sum()
            combined_remaining_missing = pd.concat([remaining_missing, (remaining_missing / total_values) * 100], axis=1)
            combined_remaining_missing.columns = ['Missing Values', 'Percentage(%)']
            display("Remaining Missing Values After Cleaning:")
            display(combined_remaining_missing[combined_remaining_missing['Missing Values'] > 0])

        display("Data cleaned successfully.")


    def analyze_promotion(self):
        train_promo = self.train_merged['Promo'].value_counts(normalize=True)
        test_promo = self.test_merged['Promo'].value_counts(normalize=True)

        plt.figure(figsize=(10, 6))
        plt.bar(['Train', 'Test'], [train_promo[1], test_promo[1]], label='Promo')
        plt.bar(['Train', 'Test'], [train_promo[0], test_promo[0]], bottom=[train_promo[1], test_promo[1]], label='No Promo')
        plt.ylabel('Proportion')
        plt.title('Distribution of Promotions in Train and Test Sets')
        plt.legend()
        plt.show()

    def analyze_holidays(self):
        holiday_map = {
            'a' : 'Public Holiday',
            'b' : 'Easter Holiday',
            'c' : 'Christmas',
            '0' : 'None'
        }

        self.train_merged['StateHoliday'] = self.train_merged['StateHoliday'].map(holiday_map)
        holiday_sales = self.train_merged.groupby('StateHoliday') ['Sales'].mean()
        plt.figure(figsize = (12 ,6))
        holiday_sales.plot(kind = 'bar')
        plt.title('Average Sales by Holiday Status')
        plt.xlabel('State Holiday')
        plt.ylabel('Average Sales')
        plt.xticks(rotation = 0)
        plt.show()

    def analyze_seasonal_behavior(self):
        monthly_sales = self.train_merged.groupby('Month')['Sales'].mean()

        plt.figure(figsize=(12, 6))
        monthly_sales.plot(kind='line', marker='o')
        plt.title('Average Monthly Sales')
        plt.xlabel('Month')
        plt.ylabel('Average Sales')
        plt.xticks(range(1, 13))
        plt.grid(True)
        plt.show()
    
    def analyze_sales_customers_correlation(self):
        correlation = self.train_merged['Sales'].corr(self.train_merged['Customers'])
        plt.figure(figsize=(10, 6))
        plt.scatter(self.train_merged['Customers'], self.train_merged['Sales'], alpha=0.5)
        plt.title(f'Sales vs Customers (Correlation: {correlation:.2f})')
        plt.xlabel('Number of Customers')
        plt.ylabel('Sales')
        plt.show()


    def analyze_promo_effect(self):
        # Overall effect of promos on sales and customers
        promo_effect = self.train_merged.groupby('Promo')[['Sales', 'Customers']].mean()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        promo_effect['Sales'].plot(kind='bar', ax=ax1)
        ax1.set_title('Average Sales with/without Promo')
        ax1.set_xlabel('Promo')
        ax1.set_ylabel('Average Sales')

        promo_effect['Customers'].plot(kind='bar', ax=ax2)
        ax2.set_title('Average Customers with/without Promo')
        ax2.set_xlabel('Promo')
        ax2.set_ylabel('Average Customers')

        plt.tight_layout()
        plt.show()

        # Effect on existing customers
        self.train_merged['SalesPerCustomer'] = self.train_merged['Sales'] / self.train_merged['Customers']
        avg_sales_per_customer = self.train_merged.groupby('Promo')['SalesPerCustomer'].mean()

        plt.figure(figsize=(10, 6))
        avg_sales_per_customer.plot(kind='bar')
        plt.title('Average Sales per Customer with/without Promo')
        plt.xlabel('Promo')
        plt.ylabel('Average Sales per Customer')
        plt.show()

        # Promo effectiveness by store type
        promo_effect_by_store = self.train_merged.groupby(['StoreType', 'Promo'])['Sales'].mean().unstack()
        promo_lift = (promo_effect_by_store[1] - promo_effect_by_store[0]) / promo_effect_by_store[0] * 100

        plt.figure(figsize=(10, 6))
        promo_lift.plot(kind='bar')
        plt.title('Promo Lift Percentage by Store Type')
        plt.xlabel('Store Type')
        plt.ylabel('Promo Lift (%)')
        plt.show()

        # Promo effectiveness by day of week
        promo_effect_by_dow = self.train_merged.groupby(['DayOfWeek', 'Promo'])['Sales'].mean().unstack()
        promo_lift_dow = (promo_effect_by_dow[1] - promo_effect_by_dow[0]) / promo_effect_by_dow[0] * 100

        plt.figure(figsize=(12, 6))
        promo_lift_dow.plot(kind='bar')
        plt.title('Promo Lift Percentage by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Promo Lift (%)')
        plt.show()

    def analyze_promo_deployment(self):
        # Calculate average sales lift for each store
        store_promo_effect = self.train_merged.groupby(['Store', 'Promo'])['Sales'].mean().unstack()
        store_promo_lift = (store_promo_effect[1] - store_promo_effect[0]) / store_promo_effect[0] * 100

        # Sort stores by promo lift
        top_stores = store_promo_lift.sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        top_stores.head(20).plot(kind='bar')
        plt.title('Top 20 Stores by Promo Lift Percentage')
        plt.xlabel('Store')
        plt.ylabel('Promo Lift (%)')
        plt.show()

        # Analyze promo effectiveness by store characteristics
        store_chars = self.store_df.set_index('Store')
        store_chars['PromoLift'] = store_promo_lift

        # Promo lift by store type
        promo_lift_by_type = store_chars.groupby('StoreType')['PromoLift'].mean()

        plt.figure(figsize=(10, 6))
        promo_lift_by_type.plot(kind='bar')
        plt.title('Average Promo Lift by Store Type')
        plt.xlabel('Store Type')
        plt.ylabel('Average Promo Lift (%)')
        plt.show()

        # Correlation of promo lift with competition distance
        plt.figure(figsize=(10, 6))
        plt.scatter(store_chars['CompetitionDistance'], store_chars['PromoLift'])
        plt.title('Promo Lift vs Competition Distance')
        plt.xlabel('Competition Distance')
        plt.ylabel('Promo Lift (%)')
        plt.show()
