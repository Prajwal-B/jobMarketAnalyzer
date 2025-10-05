# file: tests/conftest.py
"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def sample_jobs_data():
    """Create sample job postings data."""
    return pd.DataFrame({
        'job_id': [1, 2, 3, 4, 5],
        'title': [
            'Data Scientist',
            'Machine Learning Engineer',
            'Full Stack Developer',
            'DevOps Engineer',
            'Backend Developer'
        ],
        'company': [
            'TechCorp',
            'AI Solutions',
            'WebDev Inc',
            'CloudTech',
            'StartupXYZ'
        ],
        'location': [
            'Bangalore',
            'Mumbai',
            'Pune',
            'Hyderabad',
            'Delhi'
        ],
        'posted_date': [
            '2024-01-15',
            '2024-02-01',
            '2024-02-10',
            '2024-03-05',
            '2024-03-15'
        ],
        'description': [
            'Python machine learning SQL data analysis required',
            'PyTorch deep learning NLP computer vision needed',
            'React Node.js JavaScript MongoDB AWS Docker',
            'Kubernetes Jenkins Terraform Ansible Linux Bash Python',
            'Java Spring Boot PostgreSQL Redis microservices'
        ],
        'salary_min': [1000000, 1200000, 800000, 900000, 700000],
        'salary_max': [1800000, 2000000, 1500000, 1600000, 1300000],
        'experience_years': [3, 5, 3, 4, 2],
        'job_type': ['Full-time'] * 5
    })


@pytest.fixture(scope="session")
def sample_skills_lexicon():
    """Create sample skills lexicon."""
    return pd.DataFrame({
        'skill_name': [
            'Python', 'SQL', 'Machine Learning', 'JavaScript', 'React',
            'Java', 'Docker', 'Kubernetes', 'AWS', 'PostgreSQL',
            'PyTorch', 'Deep Learning', 'NLP', 'Node.js', 'MongoDB',
            'Jenkins', 'Terraform', 'Ansible', 'Linux', 'Spring Boot',
            'Redis', 'Microservices', 'Communication', 'Teamwork', 'Leadership'
        ],
        'category': [
            'Technical', 'Technical', 'Technical', 'Technical', 'Technical',
            'Technical', 'Technical', 'Technical', 'Technical', 'Technical',
            'Technical', 'Technical', 'Technical', 'Technical', 'Technical',
            'Technical', 'Technical', 'Technical', 'Technical', 'Technical',
            'Technical', 'Technical', 'Soft', 'Soft', 'Soft'
        ],
        'subcategory': [
            'Programming Language', 'Database', 'AI/ML', 'Programming Language', 'Web Development',
            'Programming Language', 'DevOps', 'DevOps', 'Cloud', 'Database',
            'AI/ML', 'AI/ML', 'AI/ML', 'Web Development', 'Database',
            'DevOps', 'DevOps', 'DevOps', 'Operating Systems', 'Web Development',
            'Database', 'Architecture', 'Interpersonal', 'Interpersonal', 'Management'
        ],
        'aliases': [
            'python,py', 'sql', 'machine learning,ml', 'javascript,js', 'react,reactjs',
            'java', 'docker', 'kubernetes,k8s', 'aws,amazon web services', 'postgresql,postgres',
            'pytorch,torch', 'deep learning,dl', 'nlp,natural language processing', 'node,nodejs', 'mongodb,mongo',
            'jenkins', 'terraform', 'ansible', 'linux,unix', 'spring boot,springboot',
            'redis', 'microservices', 'communication', 'teamwork,collaboration', 'leadership'
        ]
    })


@pytest.fixture
def sample_lexicon_file(tmp_path, sample_skills_lexicon):
    """Create sample lexicon CSV file."""
    lexicon_path = tmp_path / "skills_lexicon.csv"
    sample_skills_lexicon.to_csv(lexicon_path, index=False)
    return str(lexicon_path)


@pytest.fixture
def sample_jobs_file(tmp_path, sample_jobs_data):
    """Create sample jobs CSV file."""
    jobs_path = tmp_path / "sample_jobs.csv"
    sample_jobs_data.to_csv(jobs_path, index=False)
    return str(jobs_path)


@pytest.fixture
def processed_jobs_data(sample_jobs_data):
    """Create preprocessed jobs data with normalized fields."""
    df = sample_jobs_data.copy()
    
    # Add normalized columns
    df['title_normalized'] = df['title']
    df['location_normalized'] = df['location']
    df['description_clean'] = df['description'].str.lower()
    df['posted_date_parsed'] = pd.to_datetime(df['posted_date'])
    df['posted_year'] = df['posted_date_parsed'].dt.year
    df['posted_month'] = df['posted_date_parsed'].dt.month
    df['posted_quarter'] = df['posted_date_parsed'].dt.quarter.apply(lambda x: f'Q{x}')
    
    return df


@pytest.fixture
def jobs_with_skills_data(processed_jobs_data):
    """Create jobs data with extracted skills."""
    df = processed_jobs_data.copy()
    
    # Add extracted skills (mock)
    df['extracted_skills'] = [
        'python;machine learning;sql;data analysis',
        'pytorch;deep learning;nlp',
        'react;node.js;javascript;mongodb;aws;docker',
        'kubernetes;jenkins;terraform;ansible;linux;python',
        'java;spring boot;postgresql;redis;microservices'
    ]
    df['num_skills'] = df['extracted_skills'].apply(lambda x: len(x.split(';')))
    
    # Add boolean columns for top skills
    df['has_python'] = [True, False, False, True, False]
    df['has_sql'] = [True, False, False, False, False]
    df['has_machine_learning'] = [True, False, False, False, False]
    df['has_javascript'] = [False, False, True, False, False]
    df['has_react'] = [False, False, True, False, False]
    
    return df


# Configuration for pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require ML models"
    )


# Skip slow tests by default unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if config.getoption("-m"):
        # If marker specified, let pytest handle it
        return
    
    # Otherwise, skip slow tests by default
    skip_slow = pytest.mark.skip(reason="Slow test - use '-m slow' to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def mock_spacy_model(monkeypatch):
    """Mock spaCy model for testing without downloading."""
    class MockDoc:
        def __init__(self, text):
            self.text = text
        
        def __iter__(self):
            return iter([])
    
    class MockNLP:
        def __init__(self, name):
            self.vocab = type('obj', (object,), {'strings': {}})()
        
        def __call__(self, text):
            return MockDoc(text)
        
        def make_doc(self, text):
            return MockDoc(text)
        
        def add_pipe(self, name):
            return type('obj', (object,), {'add_label': lambda x: None})()
        
        @property
        def pipe_names(self):
            return []
    
    def mock_load(name):
        return MockNLP(name)
    
    monkeypatch.setattr("spacy.load", mock_load)


@pytest.fixture(autouse=True)
def reset_loggers():
    """Reset loggers between tests to avoid duplicate handlers."""
    import logging
    
    yield
    
    # Clear all handlers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.propagate = True


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages."""
    import logging
    caplog.set_level(logging.INFO)
    return caplog


# Environment configuration
@pytest.fixture(scope="session", autouse=True)
def set_test_environment():
    """Set up test environment variables."""
    import os
    
    # Disable warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Set random seeds for reproducibility
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    
    yield
    
    # Cleanup after all tests
    pass


# Helper fixtures for file operations
@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file path."""
    return tmp_path / "temp_data.csv"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# Performance testing fixtures
@pytest.fixture
def benchmark_data():
    """Create larger dataset for performance testing."""
    import numpy as np
    
    n_rows = 1000
    
    return pd.DataFrame({
        'job_id': range(1, n_rows + 1),
        'title': np.random.choice(['Data Scientist', 'Engineer', 'Developer'], n_rows),
        'company': [f'Company {i}' for i in range(n_rows)],
        'location': np.random.choice(['Bangalore', 'Mumbai', 'Delhi'], n_rows),
        'posted_date': pd.date_range('2024-01-01', periods=n_rows, freq='h').strftime('%Y-%m-%d'),
        'description': ['Python SQL required'] * n_rows,
        'salary_min': np.random.randint(500000, 1000000, n_rows),
        'salary_max': np.random.randint(1000000, 2000000, n_rows),
        'experience_years': np.random.randint(1, 8, n_rows)
    })