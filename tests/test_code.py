import unittest
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from unittest.mock import patch
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.customer_behaviour import CustomerBehaviour

class TestCustomerBehaviour(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sample data for testing
        cls.store_data = pd.DataFrame({
            'Store': [1, 2],
            'StoreType': ['a', 'b'],
            'Assortment': ['a', 'b'],
            'CompetitionDistance': [1000, 2000],
            'CompetitionOpenSinceMonth': [1, 2],
            'CompetitionOpenSinceYear': [2015, 2016]
        })
        cls.train_data = pd.DataFrame({
            'Store': [1, 1, 2, 2],
            'Date': ['2015-01-01', '2015-01-02', '2015-01-01', '2015-01-02'],
            'Sales': [100, 200, 150, 250],
            'Customers': [10, 20, 15, 25],
            'Open': [1, 1, 1, 1],
            'Promo': [0, 1, 0, 1],
            'StateHoliday': ['0', '0', '0', '0']
        })
        cls.test_data = pd.DataFrame({
            'Store': [1, 2],
            'Date': ['2015-01-03', '2015-01-03'],
            'Open': [1, 1],
            'Promo': [0, 1]
        })

    def setUp(self):
        # Patch the read_csv method to load sample data instead of actual CSV files
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [self.store_data, self.train_data, self.test_data]
            self.cb = CustomerBehaviour('store.csv', 'train.csv', 'test.csv')

    def test_init(self):
        # Test if initialization correctly loads the datasets
        self.assertIsInstance(self.cb.store_df, pd.DataFrame)
        self.assertIsInstance(self.cb.train_df, pd.DataFrame)
        self.assertIsInstance(self.cb.test_df, pd.DataFrame)

    def test_merge_data(self):
        # Test merging functionality
        self.cb.merge_data()
        self.assertTrue(hasattr(self.cb, 'train_merged'))
        self.assertTrue(hasattr(self.cb, 'test_merged'))
        self.assertEqual(len(self.cb.train_merged), 4)
        self.assertEqual(len(self.cb.test_merged), 2)

    @patch('matplotlib.pyplot.show')  # Prevent actual plot rendering in tests
    def test_clean_data(self, mock_show):
        # Test the data cleaning process
        self.cb.merge_data()
        self.cb.clean_data()
        self.assertFalse(self.cb.train_merged.isnull().any().any())
        self.assertFalse(self.cb.test_merged.isnull().any().any())

if __name__ == '__main__':
    unittest.main()
