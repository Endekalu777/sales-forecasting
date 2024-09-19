import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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

    def analyze_opening_hours(self):
        # Identify stores open on all weekdays
        open_stores = self.train_merged.groupby('Store')['Open'].mean()
        always_open = open_stores[open_stores == 1].index

        # Compare weekend sales for always open stores vs others
        self.train_merged['IsWeekend'] = self.train_merged['DayOfWeek'].isin([6, 7])
        weekend_sales = self.train_merged.groupby(['Store', 'IsWeekend'])['Sales'].mean().unstack()
        weekend_sales['WeekendRatio'] = weekend_sales[True] / weekend_sales[False]

        always_open_ratio = weekend_sales.loc[always_open, 'WeekendRatio'].mean()
        others_ratio = weekend_sales.loc[~weekend_sales.index.isin(always_open), 'WeekendRatio'].mean()

        plt.figure(figsize=(10, 6))
        plt.bar(['Always Open', 'Others'], [always_open_ratio, others_ratio])
        plt.title('Weekend to Weekday Sales Ratio')
        plt.ylabel('Ratio')
        plt.show()

    def analyze_assortment_effect(self):
        assortment_map = {
            'a' : 'basic',
            'b' : 'extra',
            'c' : 'extended' 
        }
        self.train_merged['Assortment'] = self.train_merged['Assortment'].map(assortment_map)
        assortment_sales = self.train_merged.groupby('Assortment')['Sales'].mean()

        plt.figure(figsize=(10, 6))
        assortment_sales.plot(kind='bar')
        plt.title('Average Sales by Assortment Type')
        plt.xlabel('Assortment Type')
        plt.ylabel('Average Sales')
        plt.show()


    def analyze_new_competitors_grouped(self):
        # Identify stores with new competitors
        new_competitor_stores = self.store_df[
            (self.store_df['CompetitionDistance'].notna()) & 
            (self.store_df['CompetitionOpenSinceYear'].notna())
        ]['Store']

        print(f"Number of stores with new competitors: {len(new_competitor_stores)}")

        if len(new_competitor_stores) == 0:
            print("No stores found with new competitors based on the current criteria.")
            return

        # Group stores by competition distance bins
        self.train_merged['CompetitionDistanceBin'] = pd.cut(
            self.train_merged['CompetitionDistance'], 
            bins=[0, 1000, 5000, 10000, np.inf], 
            labels=['<1km', '1-5km', '5-10km', '>10km']
        )

        # Analyze sales before and after new competition by distance bins
        for distance_bin in self.train_merged['CompetitionDistanceBin'].unique():
            bin_data = self.train_merged[self.train_merged['CompetitionDistanceBin'] == distance_bin]

            if bin_data.empty:
                print(f"No data for distance bin: {distance_bin}")
                continue

            # Convert year and month to datetime, handling NaN values
            competition_dates = pd.to_datetime({
                'year': bin_data['CompetitionOpenSinceYear'].fillna(2100),
                'month': bin_data['CompetitionOpenSinceMonth'].fillna(1),
                'day': 1
            }, errors='coerce')

            # Check if we have valid competition dates
            if competition_dates.isna().all():
                print(f"No valid competition dates for distance bin: {distance_bin}")
                continue

            # Split data into before and after new competition
            before_competition = bin_data[bin_data['Date'] < competition_dates]['Sales']
            after_competition = bin_data[bin_data['Date'] >= competition_dates]['Sales']

            # Check if we have data for both before and after
            if before_competition.empty or after_competition.empty:
                print(f"Insufficient data for before/after comparison in distance bin: {distance_bin}")
                continue

            before_mean = before_competition.mean()
            after_mean = after_competition.mean()

            print(f"Distance bin: {distance_bin}")
            print(f"Before competition mean: {before_mean:.2f}")
            print(f"After competition mean: {after_mean:.2f}")

            plt.figure(figsize=(10, 6))
            plt.bar(['Before Competition', 'After Competition'], [before_mean, after_mean])
            plt.title(f'Average Sales Before and After New Competition ({distance_bin} Stores)')
            plt.ylabel('Average Sales')
            plt.show()

        print("Grouped analysis of new competitors completed.")