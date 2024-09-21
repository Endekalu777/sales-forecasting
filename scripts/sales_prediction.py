import pandas as pd

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

    