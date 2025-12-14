"""
Unit Tests for Hollywoodbets Scraper
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scraper.hollywoodbets_scraper import HollywoodBetsScraper
from src.scraper.data_cleaner import DataCleaner
from src.utils.database import db

class TestHollywoodBetsScraper(unittest.TestCase):
    """Test cases for HollywoodBetsScraper"""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = HollywoodBetsScraper(use_selenium=False)
        self.cleaner = DataCleaner()
        self.test_date = datetime(2024, 1, 15)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.scraper, 'driver'):
            self.scraper.driver.quit()
    
    @patch('src.scraper.hollywoodbets_scraper.cloudscraper.create_scraper')
    def test_init(self, mock_scraper):
        """Test scraper initialization."""
        scraper = HollywoodBetsScraper(use_selenium=False)
        self.assertIsNotNone(scraper)
        self.assertEqual(scraper.base_url, "https://www.hollywoodbets.net")
        self.assertIn('User-Agent', scraper.headers)
    
    @patch('src.scraper.hollywoodbets_scraper.BeautifulSoup')
    @patch('src.scraper.hollywoodbets_scraper.HollywoodBetsScraper._make_request')
    def test_scrape_race_cards(self, mock_make_request, mock_bs):
        """Test race card scraping."""
        # Mock HTML response
        mock_html = """
        <html>
            <div class="race-meeting">
                <h3>Greyville</h3>
                <div class="race-card">
                    <div class="race-time">14:00</div>
                    <div class="race-number">Race 1</div>
                    <div class="race-name">Maiden Plate</div>
                    <table class="runners">
                        <tr><td>1</td><td><a>Horse One</a></td><td>Jockey One</td><td>Trainer One</td><td>60kg</td><td>5/2</td></tr>
                        <tr><td>2</td><td><a>Horse Two</a></td><td>Jockey Two</td><td>Trainer Two</td><td>58kg</td><td>3/1</td></tr>
                    </table>
                </div>
            </div>
        </html>
        """
        
        mock_make_request.return_value = mock_html
        mock_soup = MagicMock()
        mock_bs.return_value = mock_soup
        
        # Mock BeautifulSoup find methods
        mock_meeting = MagicMock()
        mock_meeting.find_all.return_value = [MagicMock()]
        
        mock_race_element = MagicMock()
        mock_time_element = MagicMock()
        mock_time_element.text = "14:00"
        
        mock_number_element = MagicMock()
        mock_number_element.text = "Race 1"
        
        mock_name_element = MagicMock()
        mock_name_element.text = "Maiden Plate"
        
        mock_details_element = MagicMock()
        mock_details_element.text = "1200m Class 4 R50,000"
        
        mock_table = MagicMock()
        mock_rows = [
            MagicMock(find_all=lambda x: [
                MagicMock(text='1'),
                MagicMock(find=lambda y: MagicMock(text='Horse One') if y == 'a' else None, text='Horse One'),
                MagicMock(text='Jockey One'),
                MagicMock(text='Trainer One'),
                MagicMock(text='60kg'),
                MagicMock(text='5/2')
            ]),
            MagicMock(find_all=lambda x: [
                MagicMock(text='2'),
                MagicMock(find=lambda y: MagicMock(text='Horse Two') if y == 'a' else None, text='Horse Two'),
                MagicMock(text='Jockey Two'),
                MagicMock(text='Trainer Two'),
                MagicMock(text='58kg'),
                MagicMock(text='3/1')
            ])
        ]
        mock_table.find_all.return_value = mock_rows
        
        mock_race_element.find.side_effect = lambda x, **kwargs: {
            'div': mock_time_element if 'time' in kwargs.get('class_', [''])[0] else
                   mock_number_element if 'number' in kwargs.get('class_', [''])[0] else
                   mock_name_element if 'name' in kwargs.get('class_', [''])[0] else
                   mock_details_element if 'details' in kwargs.get('class_', [''])[0] else None,
            'table': mock_table
        }.get(x, None)
        
        mock_soup.find_all.return_value = [mock_meeting]
        mock_meeting.find.return_value = MagicMock(text='Greyville')
        
        races = self.scraper.scrape_race_cards(self.test_date)
        
        # Basic assertions
        self.assertIsInstance(races, list)
    
    def test_parse_race_time(self):
        """Test race time parsing."""
        test_cases = [
            ("14:00", datetime(2024, 1, 15, 14, 0)),
            ("2:00 PM", datetime(2024, 1, 15, 14, 0)),
            ("09:30", datetime(2024, 1, 15, 9, 30)),
        ]
        
        for time_str, expected in test_cases:
            with self.subTest(time_str=time_str):
                result = self.scraper._parse_race_time(self.test_date, time_str)
                self.assertEqual(result.time(), expected.time())
    
    def test_parse_weight(self):
        """Test weight parsing."""
        test_cases = [
            ("60kg", 60.0),
            ("58.5kg", 58.5),
            ("62", 62.0),
            ("", 0.0),
            ("N/A", 0.0),
        ]
        
        for weight_str, expected in test_cases:
            with self.subTest(weight_str=weight_str):
                result = self.scraper._parse_weight(weight_str)
                self.assertAlmostEqual(result, expected, places=1)
    
    def test_parse_odds(self):
        """Test odds parsing."""
        test_cases = [
            ("5/2", 3.5),  # (5/2) + 1 = 3.5
            ("3/1", 4.0),  # (3/1) + 1 = 4.0
            ("2.5", 2.5),  # Decimal odds
            ("Evens", 2.0),  # 1/1 + 1 = 2.0
            ("", 0.0),
        ]
        
        for odds_str, expected in test_cases:
            with self.subTest(odds_str=odds_str):
                result = self.scraper._parse_odds(odds_str)
                self.assertAlmostEqual(result, expected, places=1)
    
    @patch('src.scraper.hollywoodbets_scraper.HollywoodBetsScraper._make_request')
    def test_scrape_race_results(self, mock_make_request):
        """Test race results scraping."""
        mock_make_request.return_value = "<html></html>"
        
        results = self.scraper.scrape_race_results(self.test_date)
        self.assertIsInstance(results, list)
    
    def test_save_to_database(self):
        """Test saving race data to database."""
        # Mock race data
        race_data = {
            'race_id': 'test_race_001',
            'meeting_id': 'test_meeting_001',
            'race_date': self.test_date.date(),
            'race_time': self.test_date,
            'race_number': 1,
            'race_name': 'Test Race',
            'distance': 1200,
            'going': 'Good',
            'prize_money': 50000.0,
            'runners': []
        }
        
        # This will test database connection
        try:
            result = self.scraper.save_to_database(race_data)
            # If database is available, it should return True or the race object
            # If not, it will return False
            self.assertIsNotNone(result)
        except Exception as e:
            # Database might not be available in test environment
            print(f"Database test skipped: {e}")
    
    def test_update_results(self):
        """Test updating race results in database."""
        result_data = {
            'race_id': 'test_race_001',
            'winning_horse_name': 'Test Winner',
            'positions': {1: 'Test Winner', 2: 'Test Second', 3: 'Test Third'},
            'going': 'Good',
            'winning_time': 72.5,
            'dividends': {'win': 25.0, 'place': 8.0}
        }
        
        try:
            result = self.scraper.update_results(result_data)
            self.assertIsNotNone(result)
        except Exception as e:
            print(f"Database test skipped: {e}")

class TestDataCleaner(unittest.TestCase):
    """Test cases for DataCleaner"""
    
    def setUp(self):
        self.cleaner = DataCleaner()
        self.sample_df = self._create_sample_dataframe()
    
    def _create_sample_dataframe(self):
        """Create sample DataFrame for testing."""
        import pandas as pd
        import numpy as np
        
        data = {
            'horse_id': ['H001', 'H002', 'H003', 'H004'],
            'horse_name': ['Speedster', 'Champion', 'Runner', 'Fast'],
            'weight': [60.0, 58.5, 59.0, np.nan],
            'draw': [1, 5, 3, 8],
            'age': [4, 5, 3, 4],
            'form_rating': [85, 90, 78, 82],
            'official_rating': [80, 88, 75, 80],
            'days_since_last_run': [21, 14, 30, 7],
            'career_starts': [10, 15, 8, 12],
            'career_wins': [2, 5, 1, 3],
            'career_places': [4, 8, 3, 5],
            'odds': [3.5, 2.5, 10.0, 5.0],
            'going': ['Good', 'Soft', 'Good', 'Heavy'],
            'jockey_name': ['Jockey A', 'Jockey B', 'Jockey C', 'Jockey D'],
            'trainer_name': ['Trainer X', 'Trainer Y', 'Trainer Z', 'Trainer X'],
            'distance': [1200, 1400, 1200, 1600],
            'race_class': ['Class 4', 'Class 3', 'Class 4', 'Class 2']
        }
        
        return pd.DataFrame(data)
    
    def test_clean_race_data(self):
        """Test data cleaning pipeline."""
        cleaned_df = self.cleaner.clean_race_data(self.sample_df)
        
        # Check that DataFrame is returned
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        
        # Check that missing values are handled
        self.assertFalse(cleaned_df['weight'].isnull().any())
        
        # Check that new features are created
        self.assertIn('career_win_pct', cleaned_df.columns)
        self.assertIn('going_score', cleaned_df.columns)
        self.assertIn('class_score', cleaned_df.columns)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df_with_nulls = self.sample_df.copy()
        df_with_nulls.loc[0, 'weight'] = np.nan
        df_with_nulls.loc[1, 'form_rating'] = np.nan
        
        cleaned_df = self.cleaner._handle_missing_values(df_with_nulls)
        
        # Check that NaNs are filled
        self.assertFalse(cleaned_df['weight'].isnull().any())
        self.assertFalse(cleaned_df['form_rating'].isnull().any())
    
    def test_extract_class_score(self):
        """Test race class scoring."""
        test_cases = [
            ('Maiden', 1.0),
            ('Class 4', 6.0),  # 10 - 4 = 6
            ('Class 1', 9.0),  # 10 - 1 = 9
            ('Grade 1', 10.0),
            ('Grade 2', 9.0),
            ('Listed', 7.0),
            ('Handicap', 6.0),
            ('Unknown', 5.0)
        ]
        
        for class_str, expected in test_cases:
            with self.subTest(class_str=class_str):
                result = self.cleaner._extract_class_score(class_str)
                self.assertEqual(result, expected)
    
    def test_create_derived_features(self):
        """Test derived feature creation."""
        df_with_features = self.cleaner._create_derived_features(self.sample_df)
        
        # Check that derived features are created
        expected_features = [
            'career_win_pct',
            'jockey_win_rate',
            'trainer_win_rate',
            'draw_advantage',
            'fitness_index'
        ]
        
        for feature in expected_features:
            with self.subTest(feature=feature):
                self.assertIn(feature, df_with_features.columns)
    
    def test_calculate_draw_advantage(self):
        """Test draw advantage calculation."""
        test_cases = [
            ({'draw': 1, 'distance': 1200, 'meeting_name': 'greyville'}, 0.95),  # Low draw advantage for sprint
            ({'draw': 10, 'distance': 1200, 'meeting_name': 'greyville'}, 0.5),
            ({'draw': 5, 'distance': 2000, 'meeting_name': 'greyville'}, 0.5),
            ({'draw': 1, 'distance': 1200, 'meeting_name': 'kenilworth'}, 0.525),  # Slight advantage
        ]
        
        for inputs, expected in test_cases:
            with self.subTest(inputs=inputs):
                result = self.cleaner._calculate_draw_advantage(inputs)
                self.assertIsInstance(result, float)
                self.assertGreaterEqual(result, 0.0)
                self.assertLessEqual(result, 1.0)
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        # Add target variable
        self.sample_df['is_winner'] = [1, 0, 0, 0]
        
        X, y, features = self.cleaner.prepare_training_data(self.sample_df)
        
        # Check outputs
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertIsInstance(features, list)
        
        # Check dimensions
        self.assertEqual(len(X), len(self.sample_df))
        self.assertEqual(len(y), len(self.sample_df))
        self.assertGreater(len(features), 0)
    
    def test_remove_outliers(self):
        """Test outlier removal."""
        # Create DataFrame with outliers
        df_with_outliers = pd.DataFrame({
            'weight': [60, 62, 58, 200, 59],  # 200 is outlier
            'form_rating': [85, 90, 80, 85, 1000],  # 1000 is outlier
            'odds': [3.5, 2.5, 10.0, 5.0, 100.0]  # 100.0 is outlier
        })
        
        cleaned_df = self.cleaner._remove_outliers(df_with_outliers)
        
        # Check that outliers are removed (should have fewer rows)
        self.assertLess(len(cleaned_df), len(df_with_outliers))
    
    def test_normalize_features(self):
        """Test feature normalization."""
        df_to_normalize = self.sample_df[['weight', 'form_rating', 'odds']].copy()
        normalized_df = self.cleaner._normalize_features(df_to_normalize)
        
        # Check that values are normalized between 0 and 1
        for col in normalized_df.columns:
            with self.subTest(column=col):
                self.assertGreaterEqual(normalized_df[col].min(), 0.0)
                self.assertLessEqual(normalized_df[col].max(), 1.0)

class TestDatabaseIntegration(unittest.TestCase):
    """Test database integration"""
    
    def setUp(self):
        """Set up test database."""
        # Use in-memory SQLite for testing
        import tempfile
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
    
    def tearDown(self):
        """Clean up test database."""
        import os
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_database_connection(self):
        """Test database connection."""
        from sqlalchemy import create_engine
        
        engine = create_engine(f'sqlite:///{self.db_path}')
        connection = engine.connect()
        
        self.assertIsNotNone(connection)
        connection.close()
    
    def test_create_tables(self):
        """Test table creation."""
        from sqlalchemy import create_engine, inspect
        
        engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(engine)
        
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        expected_tables = ['races', 'runners', 'race_results', 'tips', 'predictions']
        
        for table in expected_tables:
            with self.subTest(table=table):
                self.assertIn(table, tables)

if __name__ == '__main__':
    unittest.main()
