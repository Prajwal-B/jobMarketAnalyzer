# file: scripts/trend_analysis.py
"""
Time series trend analysis and anomaly detection for skills.
Analyzes monthly/quarterly trends and identifies emerging/declining skills.

Usage:
    python scripts/trend_analysis.py --input data/jobs_with_skills.csv --output exports/trends.csv
    python scripts/trend_analysis.py --input data/jobs_with_skills.csv --output exports/trends.csv --frequency monthly
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Analyze temporal trends in skill demand."""
    
    def __init__(self, frequency: str = 'monthly'):
        """
        Initialize trend analyzer.
        
        Args:
            frequency: Time aggregation frequency ('monthly', 'quarterly')
        """
        self.frequency = frequency
        self.trends_df = None
    
    def prepare_time_series(self, df: pd.DataFrame, 
                           date_column: str = 'posted_date_parsed',
                           skills_column: str = 'extracted_skills') -> pd.DataFrame:
        """
        Prepare time series data for analysis.
        
        Args:
            df: Input dataframe
            date_column: Column containing parsed dates
            skills_column: Column containing skills
            
        Returns:
            DataFrame with time series data
        """
        logger.info("Preparing time series data...")
        
        # Filter valid dates
        df_valid = df[df[date_column].notna()].copy()
        
        if len(df_valid) == 0:
            logger.error("No valid dates found in data")
            raise ValueError("No valid dates in data")
        
        logger.info(f"Valid date range: {df_valid[date_column].min()} to {df_valid[date_column].max()}")
        
        # Extract time periods
        df_valid['date'] = pd.to_datetime(df_valid[date_column])
        
        if self.frequency == 'monthly':
            df_valid['period'] = df_valid['date'].dt.to_period('M')
        elif self.frequency == 'quarterly':
            df_valid['period'] = df_valid['date'].dt.to_period('Q')
        else:
            raise ValueError(f"Unknown frequency: {self.frequency}")
        
        # Parse skills
        skill_records = []
        for _, row in df_valid.iterrows():
            period = row['period']
            skills_str = row[skills_column]
            
            if pd.notna(skills_str) and skills_str:
                skills = [s.strip() for s in str(skills_str).split(';') if s.strip()]
                for skill in skills:
                    skill_records.append({
                        'period': period,
                        'skill': skill
                    })
        
        if not skill_records:
            logger.error("No skills found in data")
            raise ValueError("No skills found")
        
        time_series_df = pd.DataFrame(skill_records)
        logger.info(f"Created time series with {len(time_series_df)} skill mentions")
        
        return time_series_df
    
    def compute_skill_trends(self, time_series_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trend metrics for each skill over time.
        
        Args:
            time_series_df: Time series dataframe
            
        Returns:
            DataFrame with trend data
        """
        logger.info("Computing skill trends...")
        
        # Count skill mentions per period
        trends = time_series_df.groupby(['period', 'skill']).size().reset_index(name='count')
        
        # Pivot to wide format
        trends_wide = trends.pivot(index='period', columns='skill', values='count').fillna(0)
        
        # Sort by period
        trends_wide = trends_wide.sort_index()
        
        logger.info(f"Computed trends for {len(trends_wide.columns)} skills over {len(trends_wide)} periods")
        
        self.trends_df = trends_wide
        
        return trends_wide
    
    def compute_growth_rates(self, trends_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute growth rates for each skill.
        
        Args:
            trends_df: Wide-format trends dataframe
            
        Returns:
            DataFrame with growth metrics
        """
        logger.info("Computing growth rates...")
        
        growth_metrics = []
        
        for skill in trends_df.columns:
            series = trends_df[skill]
            
            # Skip if no data
            if series.sum() == 0:
                continue
            
            # Compute metrics
            total_mentions = series.sum()
            mean_mentions = series.mean()
            std_mentions = series.std()
            
            # Growth rate (compare first and last periods)
            first_periods = series.head(3).mean()  # Average of first 3 periods
            last_periods = series.tail(3).mean()   # Average of last 3 periods
            
            if first_periods > 0:
                growth_rate = (last_periods - first_periods) / first_periods
            else:
                growth_rate = 0.0
            
            # Linear trend (simple linear regression)
            x = np.arange(len(series))
            y = series.values
            
            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                trend_direction = 'increasing' if slope > 0.1 else ('decreasing' if slope < -0.1 else 'stable')
            else:
                slope = 0.0
                trend_direction = 'stable'
            
            # Volatility (coefficient of variation)
            cv = std_mentions / mean_mentions if mean_mentions > 0 else 0.0
            
            growth_metrics.append({
                'skill': skill,
                'total_mentions': int(total_mentions),
                'mean_per_period': mean_mentions,
                'std_per_period': std_mentions,
                'growth_rate': growth_rate,
                'trend_slope': slope,
                'trend_direction': trend_direction,
                'volatility_cv': cv,
                'first_period_avg': first_periods,
                'last_period_avg': last_periods
            })
        
        growth_df = pd.DataFrame(growth_metrics)
        growth_df = growth_df.sort_values('growth_rate', ascending=False)
        
        return growth_df
    
    def identify_emerging_declining(self, growth_df: pd.DataFrame,
                                     top_n: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify emerging and declining skills.
        
        Args:
            growth_df: DataFrame with growth metrics
            top_n: Number of top skills to return
            
        Returns:
            Tuple of (emerging_skills, declining_skills) DataFrames
        """
        logger.info(f"Identifying top {top_n} emerging and declining skills...")
        
        # Filter skills with sufficient data
        significant_skills = growth_df[growth_df['total_mentions'] >= 5]
        
        # Emerging: high growth rate and increasing trend
        emerging = significant_skills[
            (significant_skills['growth_rate'] > 0.2) &
            (significant_skills['trend_direction'] == 'increasing')
        ].head(top_n)
        
        # Declining: negative growth rate and decreasing trend
        declining = significant_skills[
            (significant_skills['growth_rate'] < -0.2) &
            (significant_skills['trend_direction'] == 'decreasing')
        ].tail(top_n)
        
        logger.info(f"Found {len(emerging)} emerging and {len(declining)} declining skills")
        
        return emerging, declining
    
    def detect_anomalies(self, trends_df: pd.DataFrame, 
                        threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect anomalous spikes or drops in skill demand.
        
        Args:
            trends_df: Wide-format trends dataframe
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame with detected anomalies
        """
        logger.info("Detecting anomalies...")
        
        anomalies = []
        
        for skill in trends_df.columns:
            series = trends_df[skill]
            
            # Skip skills with insufficient data
            if series.sum() < 5:
                continue
            
            # Compute z-scores
            mean = series.mean()
            std = series.std()
            
            if std == 0:
                continue
            
            z_scores = (series - mean) / std
            
            # Identify anomalies
            for period, z_score in z_scores.items():
                if abs(z_score) > threshold:
                    anomalies.append({
                        'skill': skill,
                        'period': str(period),
                        'count': series[period],
                        'mean': mean,
                        'z_score': z_score,
                        'anomaly_type': 'spike' if z_score > 0 else 'drop'
                    })
        
        if anomalies:
            anomalies_df = pd.DataFrame(anomalies)
            anomalies_df = anomalies_df.sort_values('z_score', key=abs, ascending=False)
            logger.info(f"Detected {len(anomalies_df)} anomalies")
        else:
            anomalies_df = pd.DataFrame()
            logger.info("No anomalies detected")
        
        return anomalies_df
    
    def create_summary_statistics(self, trends_df: pd.DataFrame) -> Dict:
        """
        Create summary statistics for the time series.
        
        Args:
            trends_df: Wide-format trends dataframe
            
        Returns:
            Dictionary of summary statistics
        """
        total_jobs = trends_df.sum(axis=1)
        
        stats = {
            'num_periods': len(trends_df),
            'num_skills': len(trends_df.columns),
            'total_skill_mentions': int(trends_df.sum().sum()),
            'avg_jobs_per_period': total_jobs.mean(),
            'min_jobs_per_period': total_jobs.min(),
            'max_jobs_per_period': total_jobs.max(),
            'date_range_start': str(trends_df.index.min()),
            'date_range_end': str(trends_df.index.max())
        }
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze temporal trends in skill demand",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/trend_analysis.py --input data/jobs_with_skills.csv --output exports/trends.csv
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file (with extracted skills)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file (trend data)')
    parser.add_argument('--date-column', type=str, default='posted_date_parsed',
                       help='Column containing parsed dates')
    parser.add_argument('--skills-column', type=str, default='extracted_skills',
                       help='Column containing skills')
    parser.add_argument('--frequency', type=str, default='monthly',
                       choices=['monthly', 'quarterly'],
                       help='Time aggregation frequency')
    parser.add_argument('--anomaly-threshold', type=float, default=2.0,
                       help='Z-score threshold for anomaly detection')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top emerging/declining skills')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        
        # Initialize analyzer
        analyzer = TrendAnalyzer(frequency=args.frequency)
        
        # Prepare time series
        time_series_df = analyzer.prepare_time_series(
            df,
            date_column=args.date_column,
            skills_column=args.skills_column
        )
        
        # Compute trends
        trends_df = analyzer.compute_skill_trends(time_series_df)
        
        # Save trend data (long format for easier analysis)
        trends_long = trends_df.reset_index().melt(
            id_vars='period',
            var_name='skill',
            value_name='count'
        )
        trends_long['period'] = trends_long['period'].astype(str)
        
        output_path = Path(args.output)
        trends_long.to_csv(output_path, index=False)
        logger.info(f"Saved trend data to {output_path}")
        
        # Compute growth rates
        growth_df = analyzer.compute_growth_rates(trends_df)
        growth_path = output_path.parent / f"{output_path.stem}_growth.csv"
        growth_df.to_csv(growth_path, index=False)
        logger.info(f"Saved growth metrics to {growth_path}")
        
        # Identify emerging and declining skills
        emerging, declining = analyzer.identify_emerging_declining(growth_df, top_n=args.top_n)
        
        if not emerging.empty:
            emerging_path = output_path.parent / f"{output_path.stem}_emerging.csv"
            emerging.to_csv(emerging_path, index=False)
            logger.info(f"Saved emerging skills to {emerging_path}")
        
        if not declining.empty:
            declining_path = output_path.parent / f"{output_path.stem}_declining.csv"
            declining.to_csv(declining_path, index=False)
            logger.info(f"Saved declining skills to {declining_path}")
        
        # Detect anomalies
        anomalies_df = analyzer.detect_anomalies(trends_df, threshold=args.anomaly_threshold)
        
        if not anomalies_df.empty:
            anomalies_path = output_path.parent / f"{output_path.stem}_anomalies.csv"
            anomalies_df.to_csv(anomalies_path, index=False)
            logger.info(f"Saved anomalies to {anomalies_path}")
        
        # Summary statistics
        stats = analyzer.create_summary_statistics(trends_df)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("TREND ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Time range: {stats['date_range_start']} to {stats['date_range_end']}")
        logger.info(f"Number of periods: {stats['num_periods']} ({args.frequency})")
        logger.info(f"Skills analyzed: {stats['num_skills']}")
        logger.info(f"Total skill mentions: {stats['total_skill_mentions']}")
        logger.info(f"Avg jobs per period: {stats['avg_jobs_per_period']:.1f}")
        
        if not emerging.empty:
            logger.info(f"\nTop 5 Emerging Skills (highest growth):")
            for idx, row in emerging.head(5).iterrows():
                logger.info(f"  {row['skill']}: growth={row['growth_rate']:.1%}, "
                          f"mentions={row['total_mentions']}")
        
        if not declining.empty:
            logger.info(f"\nTop 5 Declining Skills (negative growth):")
            for idx, row in declining.head(5).iterrows():
                logger.info(f"  {row['skill']}: growth={row['growth_rate']:.1%}, "
                          f"mentions={row['total_mentions']}")
        
        if not anomalies_df.empty:
            logger.info(f"\nTop 5 Anomalies (by z-score):")
            for idx, row in anomalies_df.head(5).iterrows():
                logger.info(f"  {row['skill']} ({row['period']}): {row['anomaly_type']}, "
                          f"z-score={row['z_score']:.2f}")
        
        logger.info("=" * 60)
        logger.info("Trend analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()