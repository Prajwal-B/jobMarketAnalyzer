# file: scripts/utils.py
"""
Shared utility functions for the job market analyzer pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import hashlib
import json
from datetime import datetime


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging with console and optional file output.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_directories(paths: List[str]) -> None:
    """
    Ensure all required directories exist.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def validate_file_exists(filepath: str, file_type: str = "file") -> None:
    """
    Validate that a file exists, raise error if not.
    
    Args:
        filepath: Path to validate
        file_type: Type description for error message
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"{file_type} not found: {filepath}")


def compute_file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest of file hash
    """
    hash_func = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        filepath: Output file path
        indent: JSON indentation level
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def format_number(number: Union[int, float], decimal_places: int = 2) -> str:
    """
    Format number with thousands separator.
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    if isinstance(number, float):
        return f"{number:,.{decimal_places}f}"
    return f"{number:,}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value between 0 and 1 (or already percentage)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if value <= 1.0:
        value *= 100
    return f"{value:.{decimal_places}f}%"


def print_separator(char: str = "=", length: int = 60) -> None:
    """Print a separator line."""
    print(char * length)


def print_section_header(title: str, char: str = "=") -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        char: Character for border
    """
    print_separator(char)
    print(title)
    print_separator(char)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def get_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_string: strftime format string
        
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime(format_string)


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def chunks(lst: List, chunk_size: int):
    """
    Yield successive chunks from list.
    
    Args:
        lst: Input list
        chunk_size: Size of each chunk
        
    Yields:
        List chunks
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def flatten_list(nested_list: List[List]) -> List:
    """
    Flatten a nested list.
    
    Args:
        nested_list: List of lists
        
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def get_file_size(filepath: str, unit: str = "MB") -> float:
    """
    Get file size in specified unit.
    
    Args:
        filepath: Path to file
        unit: Size unit (B, KB, MB, GB)
        
    Returns:
        File size in specified unit
    """
    size_bytes = Path(filepath).stat().st_size
    
    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3
    }
    
    divisor = units.get(unit.upper(), 1)
    return size_bytes / divisor


def create_backup(filepath: str, backup_dir: Optional[str] = None) -> str:
    """
    Create a backup copy of a file.
    
    Args:
        filepath: Path to file to backup
        backup_dir: Directory for backup (optional)
        
    Returns:
        Path to backup file
    """
    import shutil
    
    source = Path(filepath)
    
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {filepath}")
    
    if backup_dir:
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
    else:
        backup_path = source.parent
    
    timestamp = get_timestamp()
    backup_file = backup_path / f"{source.stem}_backup_{timestamp}{source.suffix}"
    
    shutil.copy2(source, backup_file)
    
    return str(backup_file)


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Recursively merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_date_range(start_date: str, end_date: str, date_format: str = "%Y-%m-%d") -> bool:
    """
    Validate that start_date is before end_date.
    
    Args:
        start_date: Start date string
        end_date: End date string
        date_format: Date format string
        
    Returns:
        True if valid, False otherwise
    """
    try:
        start = datetime.strptime(start_date, date_format)
        end = datetime.strptime(end_date, date_format)
        return start <= end
    except ValueError:
        return False


def is_valid_email(email: str) -> bool:
    """
    Basic email validation.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid format, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def retry_on_exception(func, max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Decorator to retry function on exception.
    
    Args:
        func: Function to wrap
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Wrapped function
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
        return None
    
    return wrapper


def memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory stats in MB
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,
        'vms_mb': mem_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }


if __name__ == "__main__":
    # Example usage
    logger = setup_logging(log_file="logs/utils_test.log")
    logger.info("Utils module loaded successfully")
    
    # Test configuration loading
    try:
        config = load_config()
        logger.info(f"Loaded configuration: {config.get('project', {}).get('name')}")
    except FileNotFoundError:
        logger.warning("Configuration file not found")
    
    # Test utility functions
    print(f"Formatted number: {format_number(1234567.89)}")
    print(f"Formatted percentage: {format_percentage(0.8532)}")
    print(f"Timestamp: {get_timestamp()}")