# file: tests/test_preprocess.py
"""
Unit tests for preprocessing module.
Tests normalization, deduplication, date parsing, and validation.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.preprocess import (
    normalize_text,
    normalize_title,
    normalize_location,
    parse_date,
    clean_salary,
    remove_duplicates,
    validate_data,
    preprocess_jobs
)


class TestTextNormalization:
    """Test text normalization functions."""
    
    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        assert normalize_text("  HELLO  WORLD  ") == "hello world"
        assert normalize_text("Multiple   Spaces") == "multiple spaces"
    
    def test_normalize_text_empty(self):
        """Test normalization of empty/None text."""
        assert normalize_text("") == ""
        assert normalize_text(None) == ""
        assert normalize_text("   ") == ""
    
    def test_normalize_text_special_chars(self):
        """Test normalization with special characters."""
        text = "Python, SQL & JavaScript"
        result = normalize_text(text)
        assert "python" in result
        assert "sql" in result


class TestTitleNormalization:
    """Test job title normalization."""
    
    def test_normalize_data_scientist(self):
        """Test Data Scientist title variants."""
        assert normalize_title("data scientist") == "Data Scientist"
        assert normalize_title("Data Scientist") == "Data Scientist"
        assert normalize_title("DATA SCIENTIST") == "Data Scientist"
    
    def test_normalize_ml_engineer(self):
        """Test ML Engineer title variants."""
        assert normalize_title("ml engineer") == "Machine Learning Engineer"
        assert normalize_title("machine learning engineer") == "Machine Learning Engineer"
        assert normalize_title("ML Engineer") == "Machine Learning Engineer"
    
    def test_normalize_developer_roles(self):
        """Test developer role normalization."""
        assert normalize_title("full stack developer") == "Full Stack Developer"
        assert normalize_title("fullstack developer") == "Full Stack Developer"
        assert normalize_title("backend developer") == "Backend Developer"
        assert normalize_title("frontend developer") == "Frontend Developer"
    
    def test_normalize_devops(self):
        """Test DevOps role normalization."""
        assert normalize_title("devops engineer") == "DevOps Engineer"
        assert normalize_title("DevOps Engineer") == "DevOps Engineer"
    
    def test_normalize_sre(self):
        """Test SRE role normalization."""
        assert normalize_title("sre") == "Site Reliability Engineer"
        assert normalize_title("site reliability engineer") == "Site Reliability Engineer"
    
    def test_normalize_unknown_title(self):
        """Test unknown title defaults to title case."""
        result = normalize_title("random job title")
        assert result == "Random Job Title"
    
    def test_normalize_title_none(self):
        """Test None title handling."""
        assert normalize_title(None) == "Unknown"


class TestLocationNormalization:
    """Test location normalization."""
    
    def test_normalize_bangalore_variants(self):
        """Test Bangalore variants."""
        assert normalize_location("bangalore") == "Bangalore"
        assert normalize_location("Bengaluru") == "Bangalore"
        assert normalize_location("BLR") == "Bangalore"
    
    def test_normalize_mumbai_variants(self):
        """Test Mumbai variants."""
        assert normalize_location("mumbai") == "Mumbai"
        assert normalize_location("bombay") == "Mumbai"
        assert normalize_location("BOM") == "Mumbai"
    
    def test_normalize_delhi_variants(self):
        """Test Delhi variants."""
        assert normalize_location("delhi") == "Delhi"
        assert normalize_location("new delhi") == "Delhi"
        assert normalize_location("NCR") == "Delhi"
    
    def test_normalize_gurgaon(self):
        """Test Gurgaon/Gurugram variants."""
        assert normalize_location("gurgaon") == "Gurgaon"
        assert normalize_location("gurugram") == "Gurgaon"
    
    def test_normalize_hyderabad(self):
        """Test Hyderabad variants."""
        assert normalize_location("hyderabad") == "Hyderabad"
        assert normalize_location("HYD") == "Hyderabad"
    
    def test_normalize_unknown_location(self):
        """Test unknown location defaults to title case."""
        result = normalize_location("unknown city")
        assert result == "Unknown City"
    
    def test_normalize_location_none(self):
        """Test None location handling."""
        assert normalize_location(None) == "Unknown"


class TestDateParsing:
    """Test date parsing functionality."""
    
    def test_parse_standard_format(self):
        """Test standard YYYY-MM-DD format."""
        result = parse_date("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
    
    def test_parse_different_formats(self):
        """Test various date formats."""
        formats = [
            "2024-01-15",
            "15-01-2024",
            "01/15/2024",
            "15/01/2024",
            "January 15, 2024",
            "Jan 15, 2024"
        ]
        
        for date_str in formats:
            result = parse_date(date_str)
            assert result is not None
            assert isinstance(result, datetime)
    
    def test_parse_invalid_date(self):
        """Test invalid date handling."""
        result = parse_date("not a date")
        assert result is None
    
    def test_parse_none_date(self):
        """Test None date handling."""
        assert parse_date(None) is None
    
    def test_parse_empty_string(self):
        """Test empty string handling."""
        assert parse_date("") is None


class TestSalaryCleaning:
    """Test salary cleaning functionality."""
    
    def test_clean_simple_number(self):
        """Test cleaning simple numeric values."""
        assert clean_salary("1000000") == 1000000.0
        assert clean_salary(1000000) == 1000000.0
    
    def test_clean_with_commas(self):
        """Test cleaning values with commas."""
        assert clean_salary("1,000,000") == 1000000.0
    
    def test_clean_with_currency(self):
        """Test cleaning values with currency symbols."""
        assert clean_salary("₹1000000") == 1000000.0
        assert clean_salary("$1000000") == 1000000.0
    
    def test_clean_lakhs_notation(self):
        """Test lakhs notation (10L = 1,000,000)."""
        assert clean_salary("10L") == 1000000.0
        assert clean_salary("10 lakhs") == 1000000.0
    
    def test_clean_thousands_notation(self):
        """Test thousands notation (100k = 100,000)."""
        assert clean_salary("100k") == 100000.0
    
    def test_clean_range(self):
        """Test cleaning salary ranges (returns average)."""
        result = clean_salary("800000-1200000")
        assert result == 1000000.0
    
    def test_clean_none(self):
        """Test None value handling."""
        assert clean_salary(None) is None
    
    def test_clean_invalid(self):
        """Test invalid value handling."""
        assert clean_salary("invalid") is None


class TestDeduplication:
    """Test duplicate removal."""
    
    def test_remove_exact_duplicates(self):
        """Test removal of exact duplicate rows."""
        df = pd.DataFrame({
            'title': ['Data Scientist', 'Data Scientist', 'Engineer'],
            'company': ['Company A', 'Company A', 'Company B'],
            'location': ['Bangalore', 'Bangalore', 'Mumbai'],
            'posted_date': ['2024-01-01', '2024-01-01', '2024-01-02']
        })
        
        result = remove_duplicates(df)
        assert len(result) == 2
    
    def test_keep_different_dates(self):
        """Test keeping jobs with different posting dates."""
        df = pd.DataFrame({
            'title': ['Data Scientist', 'Data Scientist'],
            'company': ['Company A', 'Company A'],
            'location': ['Bangalore', 'Bangalore'],
            'posted_date': ['2024-01-01', '2024-01-02']
        })
        
        result = remove_duplicates(df)
        assert len(result) == 2
    
    def test_no_duplicates(self):
        """Test dataframe with no duplicates."""
        df = pd.DataFrame({
            'title': ['Data Scientist', 'Engineer', 'Developer'],
            'company': ['Company A', 'Company B', 'Company C'],
            'location': ['Bangalore', 'Mumbai', 'Delhi'],
            'posted_date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        
        result = remove_duplicates(df)
        assert len(result) == 3


class TestDataValidation:
    """Test data validation."""
    
    def test_remove_missing_title(self):
        """Test removal of rows with missing titles."""
        df = pd.DataFrame({
            'title': ['Data Scientist', None, 'Engineer'],
            'description': ['desc1', 'desc2', 'desc3']
        })
        
        result = validate_data(df)
        assert len(result) == 2
    
    def test_remove_missing_description(self):
        """Test removal of rows with missing descriptions."""
        df = pd.DataFrame({
            'title': ['Data Scientist', 'Engineer', 'Developer'],
            'description': ['desc1', None, '']
        })
        
        result = validate_data(df)
        assert len(result) == 1
    
    def test_validate_experience_range(self):
        """Test experience years validation."""
        df = pd.DataFrame({
            'title': ['Job1', 'Job2', 'Job3'],
            'description': ['desc1', 'desc2', 'desc3'],
            'experience_years': [-1, 5, 100]
        })
        
        result = validate_data(df)
        
        # Negative should be clipped to 0, >50 to 50
        assert result.iloc[0]['experience_years'] == 0
        assert result.iloc[1]['experience_years'] == 5
        assert result.iloc[2]['experience_years'] == 50
    
    def test_swap_inverted_salaries(self):
        """Test swapping min/max if inverted."""
        df = pd.DataFrame({
            'title': ['Job1'],
            'description': ['desc1'],
            'salary_min': [1500000],
            'salary_max': [1000000]
        })
        
        result = validate_data(df)
        
        assert result.iloc[0]['salary_min'] == 1000000
        assert result.iloc[0]['salary_max'] == 1500000


class TestFullPreprocessing:
    """Test complete preprocessing pipeline."""
    
    def test_full_pipeline(self):
        """Test full preprocessing pipeline."""
        df = pd.DataFrame({
            'job_id': [1, 2, 3],
            'title': ['data scientist', 'ml engineer', 'full stack developer'],
            'company': ['Company A', 'Company B', 'Company C'],
            'location': ['bangalore', 'mumbai', 'delhi'],
            'posted_date': ['2024-01-15', '2024-02-01', '2024-03-10'],
            'description': ['Python ML required', 'PyTorch needed', 'React and Node.js'],
            'salary_min': [1000000, 800000, 700000],
            'salary_max': [1500000, 1200000, 1100000],
            'experience_years': [3, 4, 2]
        })
        
        result = preprocess_jobs(df, remove_dups=False)
        
        # Check normalized columns exist
        assert 'title_normalized' in result.columns
        assert 'location_normalized' in result.columns
        assert 'posted_date_parsed' in result.columns
        assert 'description_clean' in result.columns
        
        # Check normalization worked
        assert result.iloc[0]['title_normalized'] == 'Data Scientist'
        assert result.iloc[0]['location_normalized'] == 'Bangalore'
        
        # Check date parsing
        assert result.iloc[0]['posted_year'] == 2024
        assert result.iloc[0]['posted_month'] == 1
    
    def test_pipeline_with_duplicates(self):
        """Test pipeline with duplicate removal."""
        df = pd.DataFrame({
            'job_id': [1, 2, 3, 4],
            'title': ['Data Scientist', 'Data Scientist', 'Engineer', 'Developer'],
            'company': ['Company A', 'Company A', 'Company B', 'Company C'],
            'location': ['Bangalore', 'Bangalore', 'Mumbai', 'Delhi'],
            'posted_date': ['2024-01-15', '2024-01-15', '2024-02-01', '2024-03-10'],
            'description': ['desc1', 'desc1', 'desc2', 'desc3'],
            'salary_min': [1000000, 1000000, 800000, 700000],
            'salary_max': [1500000, 1500000, 1200000, 1100000],
            'experience_years': [3, 3, 4, 2]
        })
        
        result = preprocess_jobs(df, remove_dups=True)
        
        # Should remove one duplicate
        assert len(result) == 3
    
    def test_pipeline_with_missing_data(self):
        """Test pipeline with missing data."""
        df = pd.DataFrame({
            'title': ['Data Scientist', None, 'Engineer'],
            'description': ['desc1', 'desc2', None],
            'location': ['Bangalore', 'Mumbai', 'Delhi'],
            'posted_date': ['2024-01-15', '2024-02-01', '2024-03-10']
        })
        
        result = preprocess_jobs(df, remove_dups=False)
        
        # Should remove rows with missing title or description
        assert len(result) == 1


@pytest.fixture
def sample_df():
    """Create sample dataframe for testing."""
    return pd.DataFrame({
        'job_id': list(range(1, 11)),
        'title': ['Data Scientist'] * 10,
        'company': [f'Company {i}' for i in range(10)],
        'location': ['Bangalore'] * 10,
        'posted_date': ['2024-01-15'] * 10,
        'description': [f'Description {i}' for i in range(10)],
        'salary_min': [1000000] * 10,
        'salary_max': [1500000] * 10,
        'experience_years': [3] * 10
    })


class TestIntegration:
    """Integration tests for preprocessing."""
    
    def test_process_and_save(self, sample_df, tmp_path):
        """Test processing and saving to file."""
        result = preprocess_jobs(sample_df, remove_dups=False)
        
        output_path = tmp_path / "processed.csv"
        result.to_csv(output_path, index=False)
        
        # Reload and verify
        reloaded = pd.read_csv(output_path)
        assert len(reloaded) == len(result)
        assert 'title_normalized' in reloaded.columns
    
    def test_chained_operations(self, sample_df):
        """Test chaining multiple preprocessing operations."""
        # Normalize
        result = preprocess_jobs(sample_df, remove_dups=False)
        
        # Further processing
        result['salary_avg'] = (result['salary_min'] + result['salary_max']) / 2
        
        assert 'salary_avg' in result.columns
        assert result['salary_avg'].iloc[0] == 1250000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])