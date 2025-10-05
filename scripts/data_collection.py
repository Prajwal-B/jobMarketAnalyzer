# file: scripts/data_collection.py
"""
Data collection module for job postings.
Supports: sample data generation, Kaggle CSV loading, web scraping templates.

Usage:
    python scripts/data_collection.py --use-sample --output data/raw_jobs.csv
    python scripts/data_collection.py --kaggle-path jobs.csv --output data/raw_jobs.csv
    python scripts/data_collection.py --scrape --max-pages 5 --output data/raw_jobs.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import random
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(output_path: str, num_rows: int = 30) -> None:
    """
    Generate synthetic sample job postings for demonstration.

    Args:
        output_path: Path to save CSV file
        num_rows: Number of sample rows to generate
    """
    logger.info(f"Generating {num_rows} sample job postings...")

    # Use existing sample_jobs.csv as template
    sample_file = Path("data/sample_jobs.csv")

    if sample_file.exists():
        logger.info(f"Loading existing sample data from {sample_file}")
        df = pd.read_csv(sample_file)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
    else:
        logger.warning("data/sample_jobs.csv not found. Creating minimal sample...")
        # Fallback: create minimal sample
        data = {
            'job_id': list(range(1, num_rows + 1)),
            'title': ['Data Scientist'] * num_rows,
            'company': ['Sample Corp'] * num_rows,
            'location': ['Bangalore'] * num_rows,
            'posted_date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                            for i in range(num_rows)],
            'description': ['Python, machine learning, SQL required'] * num_rows,
            'salary_min': [800000] * num_rows,
            'salary_max': [1500000] * num_rows,
            'experience_years': [3] * num_rows,
            'job_type': ['Full-time'] * num_rows
        }
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Created minimal sample with {num_rows} rows at {output_path}")


def load_kaggle_csv(kaggle_path: str, output_path: str,
                    chunksize: Optional[int] = None) -> None:
    """
    Load job postings from Kaggle CSV dataset.

    Args:
        kaggle_path: Path to Kaggle CSV file
        output_path: Path to save processed CSV
        chunksize: Process in chunks if dataset is large
    """
    logger.info(f"Loading Kaggle dataset from {kaggle_path}")

    if not Path(kaggle_path).exists():
        logger.error(f"Kaggle CSV not found at {kaggle_path}")
        raise FileNotFoundError(f"File not found: {kaggle_path}")

    try:
        if chunksize:
            logger.info(f"Processing in chunks of {chunksize} rows...")
            chunks = []
            for chunk in pd.read_csv(kaggle_path, chunksize=chunksize):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(kaggle_path)

        logger.info(f"Loaded {len(df)} rows from Kaggle dataset")

        # Basic column mapping (adjust based on actual Kaggle dataset structure)
        column_mapping = {
            'Job Title': 'title',
            'Company Name': 'company',
            'Location': 'location',
            'Date Posted': 'posted_date',
            'Job Description': 'description',
            'Min Salary': 'salary_min',
            'Max Salary': 'salary_max',
            'Experience': 'experience_years',
            'Employment Type': 'job_type'
        }

        # Rename columns if they exist
        existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        if existing_cols:
            df = df.rename(columns=existing_cols)

        # Add job_id if not present
        if 'job_id' not in df.columns:
            df.insert(0, 'job_id', range(1, len(df) + 1))

        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")

    except Exception as e:
        logger.error(f"Error loading Kaggle CSV: {e}")
        raise


def scrape_jobs_template(max_pages: int = 5, output_path: str = "data/scraped_jobs.csv") -> None:
    """
    Template for web scraping job postings.

    WARNING: This is a TEMPLATE only. Before scraping any website:
    1. Check robots.txt file
    2. Review terms of service
    3. Implement rate limiting
    4. Respect website policies

    Args:
        max_pages: Maximum number of pages to scrape
        output_path: Path to save scraped data
    """
    logger.warning("=" * 80)
    logger.warning("WEB SCRAPING TEMPLATE - READ CAREFULLY")
    logger.warning("=" * 80)
    logger.warning("This is a demonstration template only.")
    logger.warning("Before scraping any website:")
    logger.warning("  1. Check robots.txt (e.g., https://example.com/robots.txt)")
    logger.warning("  2. Review Terms of Service")
    logger.warning("  3. Implement proper rate limiting (delays between requests)")
    logger.warning("  4. Use respectful User-Agent headers")
    logger.warning("  5. Consider using official APIs instead")
    logger.warning("=" * 80)

    # Example template (does not scrape real sites)
    logger.info("Scraping template - generating mock data instead of real scraping")

    jobs = []

    for page in range(1, min(max_pages, 3) + 1):
        logger.info(f"Processing page {page}/{max_pages}")

        # Mock data generation (replace with actual scraping logic)
        for i in range(10):
            job = {
                'job_id': len(jobs) + 1,
                'title': f'Sample Job {len(jobs) + 1}',
                'company': f'Company {random.randint(1, 50)}',
                'location': random.choice(['Bangalore', 'Mumbai', 'Delhi', 'Pune', 'Hyderabad']),
                'posted_date': (datetime.now() - timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d'),
                'description': 'Python, SQL, machine learning, data analysis required',
                'salary_min': random.randint(500000, 1000000),
                'salary_max': random.randint(1000000, 2000000),
                'experience_years': random.randint(1, 7),
                'job_type': 'Full-time'
            }
            jobs.append(job)

        # Rate limiting - ALWAYS implement this for real scraping
        time.sleep(2)

    df = pd.DataFrame(jobs)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} scraped jobs to {output_path}")

    # Example of how to structure real scraping (commented out)
    """
    # Real scraping example (DO NOT USE without permission):
    headers = {
        'User-Agent': 'JobAnalyzer/1.0 (Educational Research; contact@example.com)'
    }

    for page in range(1, max_pages + 1):
        url = f"https://example-job-site.com/jobs?page={page}"

        # Always check robots.txt first
        # Always implement error handling
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract job listings (adjust selectors for target site)
            job_cards = soup.find_all('div', class_='job-card')

            for card in job_cards:
                job = {
                    'title': card.find('h2', class_='title').text.strip(),
                    'company': card.find('span', class_='company').text.strip(),
                    # ... extract other fields
                }
                jobs.append(job)

            # CRITICAL: Rate limiting (2-5 seconds between requests)
            time.sleep(3)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping page {page}: {e}")
            continue
    """


def main():
    parser = argparse.ArgumentParser(
        description="Job data collection module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample data (default)
  python scripts/data_collection.py --use-sample --output data/raw_jobs.csv

  # Load from Kaggle CSV
  python scripts/data_collection.py --kaggle-path ~/downloads/jobs.csv --output data/raw_jobs.csv

  # Scraping template (mock data)
  python scripts/data_collection.py --scrape --max-pages 3 --output data/scraped_jobs.csv
        """
    )

    parser.add_argument('--use-sample', action='store_true',
                        help='Generate sample data for demonstration')
    parser.add_argument('--kaggle-path', type=str,
                        help='Path to Kaggle CSV dataset')
    parser.add_argument('--scrape', action='store_true',
                        help='Use web scraping template (generates mock data)')
    parser.add_argument('--max-pages', type=int, default=5,
                        help='Maximum pages to scrape (default: 5)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--num-rows', type=int, default=30,
                        help='Number of sample rows (for --use-sample)')
    parser.add_argument('--chunksize', type=int,
                        help='Process large CSVs in chunks')

    args = parser.parse_args()

    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.use_sample:
            generate_sample_data(args.output, args.num_rows)
        elif args.kaggle_path:
            load_kaggle_csv(args.kaggle_path, args.output, args.chunksize)
        elif args.scrape:
            scrape_jobs_template(args.max_pages, args.output)
        else:
            logger.error("No data source specified. Use --use-sample, --kaggle-path, or --scrape")
            parser.print_help()
            sys.exit(1)

        logger.info("Data collection completed successfully!")

    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()