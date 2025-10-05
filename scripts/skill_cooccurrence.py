# file: scripts/skill_cooccurrence.py
"""
Skill co-occurrence analysis and network generation.
Creates co-occurrence matrix, network graphs, and association rules.

Usage:
    python scripts/skill_cooccurrence.py --input data/jobs_with_skills.csv --output exports/cooccurrence.csv
    python scripts/skill_cooccurrence.py --input data/jobs_with_skills.csv --output exports/cooccurrence.csv --min-support 0.05
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SkillCooccurrenceAnalyzer:
    """Analyze skill co-occurrence patterns and build network graphs."""
    
    def __init__(self, min_support: float = 0.02, min_confidence: float = 0.3):
        """
        Initialize analyzer.
        
        Args:
            min_support: Minimum support threshold for association rules
            min_confidence: Minimum confidence threshold for association rules
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.cooccurrence_matrix = None
        self.network_graph = None
    
    def parse_skills(self, df: pd.DataFrame, skills_column: str = 'extracted_skills') -> List[Set[str]]:
        """
        Parse skills from dataframe into list of skill sets.
        
        Args:
            df: Input dataframe
            skills_column: Column containing semicolon-separated skills
            
        Returns:
            List of skill sets
        """
        skill_sets = []
        
        for skills_str in df[skills_column]:
            if pd.notna(skills_str) and skills_str:
                skills = set(s.strip() for s in str(skills_str).split(';') if s.strip())
                skill_sets.append(skills)
            else:
                skill_sets.append(set())
        
        return skill_sets
    
    def build_cooccurrence_matrix(self, skill_sets: List[Set[str]]) -> pd.DataFrame:
        """
        Build skill co-occurrence matrix.
        
        Args:
            skill_sets: List of skill sets from job postings
            
        Returns:
            DataFrame with co-occurrence counts
        """
        logger.info("Building co-occurrence matrix...")
        
        # Count co-occurrences
        cooccurrence = defaultdict(lambda: defaultdict(int))
        skill_counts = defaultdict(int)
        
        for skills in skill_sets:
            # Count individual skills
            for skill in skills:
                skill_counts[skill] += 1
            
            # Count pairs
            for skill1, skill2 in combinations(sorted(skills), 2):
                cooccurrence[skill1][skill2] += 1
                cooccurrence[skill2][skill1] += 1  # Symmetric
        
        # Convert to dataframe
        all_skills = sorted(skill_counts.keys())
        matrix = pd.DataFrame(0, index=all_skills, columns=all_skills)
        
        for skill1 in all_skills:
            for skill2 in all_skills:
                if skill1 != skill2:
                    matrix.loc[skill1, skill2] = cooccurrence[skill1].get(skill2, 0)
        
        self.cooccurrence_matrix = matrix
        self.skill_counts = skill_counts
        
        logger.info(f"Created {len(all_skills)} x {len(all_skills)} co-occurrence matrix")
        
        return matrix
    
    def compute_association_metrics(self, skill_sets: List[Set[str]]) -> pd.DataFrame:
        """
        Compute association metrics (lift, confidence) for skill pairs.
        
        Args:
            skill_sets: List of skill sets
            
        Returns:
            DataFrame with association metrics
        """
        logger.info("Computing association metrics...")
        
        total_jobs = len(skill_sets)
        associations = []
        
        # Get skill pairs with sufficient support
        for skill1, skill2 in combinations(sorted(self.skill_counts.keys()), 2):
            # Count co-occurrences
            cooccur_count = sum(1 for skills in skill_sets if skill1 in skills and skill2 in skills)
            
            if cooccur_count == 0:
                continue
            
            # Calculate metrics
            support = cooccur_count / total_jobs
            
            if support < self.min_support:
                continue
            
            skill1_count = self.skill_counts[skill1]
            skill2_count = self.skill_counts[skill2]
            
            # Confidence: P(skill2 | skill1)
            confidence_1_to_2 = cooccur_count / skill1_count
            confidence_2_to_1 = cooccur_count / skill2_count
            
            # Lift: (P(skill1 & skill2)) / (P(skill1) * P(skill2))
            expected = (skill1_count / total_jobs) * (skill2_count / total_jobs) * total_jobs
            lift = cooccur_count / expected if expected > 0 else 0
            
            associations.append({
                'skill1': skill1,
                'skill2': skill2,
                'cooccurrence_count': cooccur_count,
                'support': support,
                'confidence_1_to_2': confidence_1_to_2,
                'confidence_2_to_1': confidence_2_to_1,
                'lift': lift,
                'skill1_count': skill1_count,
                'skill2_count': skill2_count
            })
        
        df = pd.DataFrame(associations)
        
        if not df.empty:
            # Sort by lift
            df = df.sort_values('lift', ascending=False)
            logger.info(f"Found {len(df)} significant skill associations")
        else:
            logger.warning("No significant associations found")
        
        return df
    
    def build_network_graph(self, associations_df: pd.DataFrame, 
                           top_n: int = 100) -> nx.Graph:
        """
        Build network graph from skill associations.
        
        Args:
            associations_df: DataFrame with association metrics
            top_n: Number of top associations to include
            
        Returns:
            NetworkX graph
        """
        logger.info(f"Building network graph with top {top_n} associations...")
        
        G = nx.Graph()
        
        # Take top N associations by lift
        top_associations = associations_df.head(top_n)
        
        for _, row in top_associations.iterrows():
            skill1 = row['skill1']
            skill2 = row['skill2']
            
            # Add nodes with attributes
            if not G.has_node(skill1):
                G.add_node(skill1, count=row['skill1_count'])
            if not G.has_node(skill2):
                G.add_node(skill2, count=row['skill2_count'])
            
            # Add edge with attributes
            G.add_edge(
                skill1, skill2,
                weight=row['cooccurrence_count'],
                lift=row['lift'],
                support=row['support'],
                confidence=max(row['confidence_1_to_2'], row['confidence_2_to_1'])
            )
        
        self.network_graph = G
        
        logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def compute_network_metrics(self, G: nx.Graph) -> pd.DataFrame:
        """
        Compute centrality and other network metrics.
        
        Args:
            G: NetworkX graph
            
        Returns:
            DataFrame with node metrics
        """
        logger.info("Computing network metrics...")
        
        metrics = []
        
        # Compute centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        try:
            pagerank = nx.pagerank(G, weight='weight')
        except:
            pagerank = {node: 0.0 for node in G.nodes()}
        
        for node in G.nodes():
            metrics.append({
                'skill': node,
                'degree': G.degree(node),
                'degree_centrality': degree_centrality[node],
                'betweenness_centrality': betweenness_centrality[node],
                'closeness_centrality': closeness_centrality[node],
                'pagerank': pagerank[node],
                'count': G.nodes[node].get('count', 0)
            })
        
        df = pd.DataFrame(metrics)
        df = df.sort_values('degree', ascending=False)
        
        return df
    
    def identify_skill_clusters(self, G: nx.Graph) -> Dict[str, int]:
        """
        Identify skill clusters/communities.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary mapping skills to cluster IDs
        """
        logger.info("Identifying skill clusters...")
        
        # Use community detection
        from networkx.algorithms import community
        
        try:
            communities = community.greedy_modularity_communities(G, weight='weight')
            
            skill_to_cluster = {}
            for cluster_id, cluster_nodes in enumerate(communities):
                for node in cluster_nodes:
                    skill_to_cluster[node] = cluster_id
            
            logger.info(f"Identified {len(communities)} skill clusters")
            
            return skill_to_cluster
        
        except Exception as e:
            logger.warning(f"Could not identify clusters: {e}")
            return {}
    
    def export_network(self, G: nx.Graph, output_path: str, format: str = 'graphml'):
        """
        Export network graph to file.
        
        Args:
            G: NetworkX graph
            output_path: Output file path
            format: Export format (graphml, gexf, edgelist)
        """
        logger.info(f"Exporting network to {output_path} (format: {format})")
        
        output_path = Path(output_path)
        
        if format == 'graphml':
            nx.write_graphml(G, output_path)
        elif format == 'gexf':
            nx.write_gexf(G, output_path)
        elif format == 'edgelist':
            nx.write_edgelist(G, output_path, data=True)
        else:
            logger.warning(f"Unknown format: {format}, using graphml")
            nx.write_graphml(G, output_path.with_suffix('.graphml'))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze skill co-occurrence and build network graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/skill_cooccurrence.py --input data/jobs_with_skills.csv --output exports/cooccurrence.csv
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file (with extracted skills)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file (co-occurrence matrix)')
    parser.add_argument('--skills-column', type=str, default='extracted_skills',
                       help='Column containing skills')
    parser.add_argument('--min-support', type=float, default=0.02,
                       help='Minimum support for associations (0.0-1.0)')
    parser.add_argument('--min-confidence', type=float, default=0.3,
                       help='Minimum confidence for associations (0.0-1.0)')
    parser.add_argument('--top-n', type=int, default=100,
                       help='Top N associations for network graph')
    parser.add_argument('--network-format', type=str, default='graphml',
                       choices=['graphml', 'gexf', 'edgelist'],
                       help='Network export format')
    
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
        
        if args.skills_column not in df.columns:
            logger.error(f"Skills column '{args.skills_column}' not found in data")
            sys.exit(1)
        
        # Initialize analyzer
        analyzer = SkillCooccurrenceAnalyzer(
            min_support=args.min_support,
            min_confidence=args.min_confidence
        )
        
        # Parse skills
        skill_sets = analyzer.parse_skills(df, args.skills_column)
        logger.info(f"Parsed skills from {len(skill_sets)} job postings")
        
        # Build co-occurrence matrix
        matrix = analyzer.build_cooccurrence_matrix(skill_sets)
        
        # Save co-occurrence matrix
        matrix_path = Path(args.output)
        matrix.to_csv(matrix_path)
        logger.info(f"Saved co-occurrence matrix to {matrix_path}")
        
        # Compute association metrics
        associations_df = analyzer.compute_association_metrics(skill_sets)
        
        if not associations_df.empty:
            # Save associations
            assoc_path = matrix_path.parent / f"{matrix_path.stem}_associations.csv"
            associations_df.to_csv(assoc_path, index=False)
            logger.info(f"Saved association metrics to {assoc_path}")
            
            # Build network graph
            G = analyzer.build_network_graph(associations_df, top_n=args.top_n)
            
            # Save network
            network_path = matrix_path.parent / f"skill_network.{args.network_format}"
            analyzer.export_network(G, str(network_path), format=args.network_format)
            
            # Compute and save network metrics
            node_metrics = analyzer.compute_network_metrics(G)
            metrics_path = matrix_path.parent / "network_metrics.csv"
            node_metrics.to_csv(metrics_path, index=False)
            logger.info(f"Saved network metrics to {metrics_path}")
            
            # Identify clusters
            clusters = analyzer.identify_skill_clusters(G)
            if clusters:
                cluster_df = pd.DataFrame([
                    {'skill': skill, 'cluster': cluster_id}
                    for skill, cluster_id in clusters.items()
                ])
                cluster_path = matrix_path.parent / "skill_clusters.csv"
                cluster_df.to_csv(cluster_path, index=False)
                logger.info(f"Saved skill clusters to {cluster_path}")
            
            # Print summary
            logger.info("=" * 60)
            logger.info("CO-OCCURRENCE ANALYSIS SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total skills analyzed: {len(matrix)}")
            logger.info(f"Significant associations: {len(associations_df)}")
            logger.info(f"Network nodes: {G.number_of_nodes()}")
            logger.info(f"Network edges: {G.number_of_edges()}")
            
            # Top associations
            logger.info("\nTop 10 Skill Pairs (by lift):")
            for idx, row in associations_df.head(10).iterrows():
                logger.info(f"  {row['skill1']} <-> {row['skill2']}: "
                          f"lift={row['lift']:.2f}, support={row['support']:.3f}")
            
            # Top central skills
            logger.info("\nTop 10 Central Skills (by degree):")
            for idx, row in node_metrics.head(10).iterrows():
                logger.info(f"  {row['skill']}: degree={row['degree']}, "
                          f"count={row['count']}")
            
            logger.info("=" * 60)
        
        logger.info("Co-occurrence analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Co-occurrence analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()