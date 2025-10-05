# file: scripts/geography_analysis.py
"""
Geographic analysis of job postings and skills.
Maps locations to Tier-1/Tier-2 cities and generates regional breakdowns.

Usage:
    python scripts/geography_analysis.py --input data/jobs_with_skills.csv --output-dir exports/by_region/
    python scripts/geography_analysis.py --input data/jobs_with_skills.csv --output-dir exports/by_region/ --export-tableau
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Indian city tier classification
CITY_TIERS = {
    'Tier-1': [
        'Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 
        'Kolkata', 'Pune', 'Ahmedabad'
    ],
    'Tier-2': [
        'Jaipur', 'Surat', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore',
        'Thane', 'Bhopal', 'Visakhapatnam', 'Patna', 'Vadodara',
        'Ghaziabad', 'Ludhiana', 'Coimbatore', 'Kochi', 'Gurgaon',
        'Noida', 'Chandigarh', 'Agra', 'Madurai', 'Nashik', 'Varanasi'
    ]
}


# Regional groupings
REGIONS = {
    'North': ['Delhi', 'Gurgaon', 'Noida', 'Ghaziabad', 'Chandigarh', 'Lucknow', 
              'Kanpur', 'Jaipur', 'Ludhiana', 'Agra', 'Varanasi'],
    'South': ['Bangalore', 'Hyderabad', 'Chennai', 'Kochi', 'Coimbatore', 
              'Visakhapatnam', 'Madurai'],
    'West': ['Mumbai', 'Pune', 'Ahmedabad', 'Surat', 'Thane', 'Vadodara', 'Nashik'],
    'East': ['Kolkata', 'Patna'],
    'Central': ['Bhopal', 'Nagpur', 'Indore']
}


class GeographyAnalyzer:
    """Analyze geographic distribution of jobs and skills."""
    
    def __init__(self):
        """Initialize geography analyzer."""
        self.city_to_tier = self._build_city_tier_mapping()
        self.city_to_region = self._build_city_region_mapping()
    
    def _build_city_tier_mapping(self) -> Dict[str, str]:
        """Build mapping from city to tier."""
        mapping = {}
        for tier, cities in CITY_TIERS.items():
            for city in cities:
                mapping[city.lower()] = tier
        return mapping
    
    def _build_city_region_mapping(self) -> Dict[str, str]:
        """Build mapping from city to region."""
        mapping = {}
        for region, cities in REGIONS.items():
            for city in cities:
                mapping[city.lower()] = region
        return mapping
    
    def enrich_locations(self, df: pd.DataFrame, 
                        location_column: str = 'location_normalized') -> pd.DataFrame:
        """
        Add tier and region information to dataframe.
        
        Args:
            df: Input dataframe
            location_column: Column containing normalized location
            
        Returns:
            Enriched dataframe
        """
        logger.info("Enriching location data with tier and region...")
        
        df = df.copy()
        
        # Add tier
        df['city_tier'] = df[location_column].apply(
            lambda x: self.city_to_tier.get(str(x).lower(), 'Other') if pd.notna(x) else 'Unknown'
        )
        
        # Add region
        df['region'] = df[location_column].apply(
            lambda x: self.city_to_region.get(str(x).lower(), 'Other') if pd.notna(x) else 'Unknown'
        )
        
        logger.info(f"Tier distribution: {df['city_tier'].value_counts().to_dict()}")
        logger.info(f"Region distribution: {df['region'].value_counts().to_dict()}")
        
        return df
    
    def analyze_by_location(self, df: pd.DataFrame,
                           location_column: str = 'location_normalized') -> pd.DataFrame:
        """
        Aggregate statistics by location.
        
        Args:
            df: Input dataframe
            location_column: Column containing location
            
        Returns:
            DataFrame with location statistics
        """
        logger.info("Analyzing by location...")
        
        location_stats = df.groupby(location_column).agg({
            'job_id': 'count',
            'num_skills': 'mean',
            'salary_min': 'mean',
            'salary_max': 'mean',
            'experience_years': 'mean'
        }).reset_index()
        
        location_stats.columns = [
            'location', 'job_count', 'avg_skills', 
            'avg_salary_min', 'avg_salary_max', 'avg_experience'
        ]
        
        # Add tier and region
        location_stats['tier'] = location_stats['location'].apply(
            lambda x: self.city_to_tier.get(str(x).lower(), 'Other')
        )
        location_stats['region'] = location_stats['location'].apply(
            lambda x: self.city_to_region.get(str(x).lower(), 'Other')
        )
        
        location_stats = location_stats.sort_values('job_count', ascending=False)
        
        return location_stats
    
    def analyze_skills_by_location(self, df: pd.DataFrame,
                                   location_column: str = 'location_normalized',
                                   skills_column: str = 'extracted_skills',
                                   top_n: int = 20) -> pd.DataFrame:
        """
        Analyze top skills by location.
        
        Args:
            df: Input dataframe
            location_column: Column containing location
            skills_column: Column containing skills
            top_n: Number of top skills per location
            
        Returns:
            DataFrame with location-skill statistics
        """
        logger.info("Analyzing skills by location...")
        
        skill_location_data = []
        
        for location in df[location_column].unique():
            if pd.isna(location) or location == 'Unknown':
                continue
            
            location_df = df[df[location_column] == location]
            
            # Count skills
            skill_counts = {}
            for skills_str in location_df[skills_column]:
                if pd.notna(skills_str) and skills_str:
                    skills = [s.strip() for s in str(skills_str).split(';') if s.strip()]
                    for skill in skills:
                        skill_counts[skill] = skill_counts.get(skill, 0) + 1
            
            # Get top skills
            sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
            
            for skill, count in sorted_skills[:top_n]:
                skill_location_data.append({
                    'location': location,
                    'skill': skill,
                    'count': count,
                    'percentage': count / len(location_df) * 100
                })
        
        skill_location_df = pd.DataFrame(skill_location_data)
        
        return skill_location_df
    
    def analyze_by_tier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate statistics by city tier.
        
        Args:
            df: Input dataframe with city_tier column
            
        Returns:
            DataFrame with tier statistics
        """
        logger.info("Analyzing by city tier...")
        
        tier_stats = df.groupby('city_tier').agg({
            'job_id': 'count',
            'num_skills': 'mean',
            'salary_min': 'mean',
            'salary_max': 'mean',
            'experience_years': 'mean'
        }).reset_index()
        
        tier_stats.columns = [
            'tier', 'job_count', 'avg_skills',
            'avg_salary_min', 'avg_salary_max', 'avg_experience'
        ]
        
        tier_stats = tier_stats.sort_values('job_count', ascending=False)
        
        return tier_stats
    
    def analyze_by_region(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate statistics by region.
        
        Args:
            df: Input dataframe with region column
            
        Returns:
            DataFrame with region statistics
        """
        logger.info("Analyzing by region...")
        
        region_stats = df.groupby('region').agg({
            'job_id': 'count',
            'num_skills': 'mean',
            'salary_min': 'mean',
            'salary_max': 'mean',
            'experience_years': 'mean'
        }).reset_index()
        
        region_stats.columns = [
            'region', 'job_count', 'avg_skills',
            'avg_salary_min', 'avg_salary_max', 'avg_experience'
        ]
        
        region_stats = region_stats.sort_values('job_count', ascending=False)
        
        return region_stats
    
    def export_regional_subsets(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Export separate CSV files for each region.
        
        Args:
            df: Input dataframe
            output_dir: Output directory
        """
        logger.info("Exporting regional subsets...")
        
        for region in df['region'].unique():
            if region == 'Unknown':
                continue
            
            region_df = df[df['region'] == region]
            
            if len(region_df) > 0:
                filename = f"jobs_{region.lower()}.csv"
                filepath = output_dir / filename
                region_df.to_csv(filepath, index=False)
                logger.info(f"Saved {len(region_df)} jobs for {region} to {filepath}")
    
    def create_tableau_export(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        Create Tableau-ready export with geographic data.
        
        Args:
            df: Input dataframe
            output_path: Output file path
        """
        logger.info("Creating Tableau export...")
        
        # Select relevant columns for Tableau
        tableau_cols = [
            'job_id', 'title', 'company', 'location_normalized',
            'city_tier', 'region', 'posted_date', 'posted_year', 'posted_month',
            'salary_min', 'salary_max', 'experience_years', 'num_skills'
        ]
        
        # Add boolean skill columns
        skill_cols = [col for col in df.columns if col.startswith('has_')]
        tableau_cols.extend(skill_cols)
        
        # Filter existing columns
        available_cols = [col for col in tableau_cols if col in df.columns]
        
        tableau_df = df[available_cols].copy()
        
        # Rename for clarity
        rename_map = {
            'location_normalized': 'location',
            'posted_year': 'year',
            'posted_month': 'month'
        }
        tableau_df = tableau_df.rename(columns=rename_map)
        
        tableau_df.to_csv(output_path, index=False)
        logger.info(f"Saved Tableau export to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze geographic distribution of jobs and skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/geography_analysis.py --input data/jobs_with_skills.csv --output-dir exports/by_region/
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file (with extracted skills)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for regional exports')
    parser.add_argument('--location-column', type=str, default='location_normalized',
                       help='Column containing normalized location')
    parser.add_argument('--skills-column', type=str, default='extracted_skills',
                       help='Column containing skills')
    parser.add_argument('--export-tableau', action='store_true',
                       help='Create Tableau-ready export')
    parser.add_argument('--top-skills', type=int, default=20,
                       help='Number of top skills per location')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        
        # Initialize analyzer
        analyzer = GeographyAnalyzer()
        
        # Enrich with tier and region
        df = analyzer.enrich_locations(df, args.location_column)
        
        # Location analysis
        location_stats = analyzer.analyze_by_location(df, args.location_column)
        location_path = output_dir / "location_statistics.csv"
        location_stats.to_csv(location_path, index=False)
        logger.info(f"Saved location statistics to {location_path}")
        
        # Tier analysis
        tier_stats = analyzer.analyze_by_tier(df)
        tier_path = output_dir / "tier_statistics.csv"
        tier_stats.to_csv(tier_path, index=False)
        logger.info(f"Saved tier statistics to {tier_path}")
        
        # Region analysis
        region_stats = analyzer.analyze_by_region(df)
        region_path = output_dir / "region_statistics.csv"
        region_stats.to_csv(region_path, index=False)
        logger.info(f"Saved region statistics to {region_path}")
        
        # Skills by location
        skill_location_df = analyzer.analyze_skills_by_location(
            df, args.location_column, args.skills_column, args.top_skills
        )
        skill_location_path = output_dir / "skills_by_location.csv"
        skill_location_df.to_csv(skill_location_path, index=False)
        logger.info(f"Saved skills by location to {skill_location_path}")
        
        # Export regional subsets
        analyzer.export_regional_subsets(df, output_dir)
        
        # Tableau export
        if args.export_tableau:
            tableau_path = output_dir.parent / "tableau_ready" / "jobs_geographic.csv"
            tableau_path.parent.mkdir(parents=True, exist_ok=True)
            analyzer.create_tableau_export(df, tableau_path)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("GEOGRAPHIC ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total jobs analyzed: {len(df)}")
        logger.info(f"Unique locations: {df[args.location_column].nunique()}")
        
        logger.info("\nJobs by Tier:")
        for _, row in tier_stats.iterrows():
            logger.info(f"  {row['tier']}: {row['job_count']} jobs "
                       f"(avg skills: {row['avg_skills']:.1f})")
        
        logger.info("\nJobs by Region:")
        for _, row in region_stats.iterrows():
            logger.info(f"  {row['region']}: {row['job_count']} jobs")
        
        logger.info("\nTop 5 Cities by Job Count:")
        for idx, row in location_stats.head(5).iterrows():
            logger.info(f"  {row['location']}: {row['job_count']} jobs "
                       f"({row['tier']}, {row['region']})")
        
        logger.info("=" * 60)
        logger.info("Geographic analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Geographic analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()