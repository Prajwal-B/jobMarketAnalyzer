# file: scripts/skill_extractor.py
"""
Hybrid skill extraction module.
Combines: spaCy PhraseMatcher (rule-based) + Sentence-BERT (embedding-based).

Usage:
    python scripts/skill_extractor.py --input data/cleaned_jobs.csv --output data/jobs_with_skills.csv
    python scripts/skill_extractor.py --input data/cleaned_jobs.csv --output data/jobs_with_skills.csv --use-embeddings
    python scripts/skill_extractor.py --input data/cleaned_jobs.csv --output data/jobs_with_skills.csv --use-embeddings --similarity-threshold 0.80
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Set, Dict, Tuple
import pandas as pd
import numpy as np
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SkillExtractor:
    """Hybrid skill extraction using PhraseMatcher and embeddings."""
    
    def __init__(self, lexicon_path: str, use_embeddings: bool = False,
                 similarity_threshold: float = 0.78, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize skill extractor.
        
        Args:
            lexicon_path: Path to skills lexicon CSV
            use_embeddings: Whether to use embedding-based matching
            similarity_threshold: Cosine similarity threshold for embeddings
            model_name: Sentence-BERT model name
        """
        logger.info("Initializing SkillExtractor...")
        
        # Load lexicon
        self.lexicon_df = pd.read_csv(lexicon_path)
        logger.info(f"Loaded {len(self.lexicon_df)} skills from lexicon")
        
        # Create skill mapping (canonical name -> aliases)
        self.skill_mapping = self._build_skill_mapping()
        
        # Load spaCy model
        logger.info("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        # Initialize PhraseMatcher
        self.phrase_matcher = self._build_phrase_matcher()
        logger.info(f"Initialized PhraseMatcher with {len(self.skill_mapping)} skill patterns")
        
        # Initialize embeddings if requested
        self.use_embeddings = use_embeddings
        self.similarity_threshold = similarity_threshold
        self.embedding_model = None
        self.skill_embeddings = None
        
        if use_embeddings:
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            self._precompute_skill_embeddings()
            logger.info("Embedding model loaded and skill embeddings precomputed")
    
    def _build_skill_mapping(self) -> Dict[str, List[str]]:
        """
        Build mapping of canonical skill names to all possible aliases.
        
        Returns:
            Dictionary mapping canonical names to list of variants
        """
        skill_mapping = {}
        
        for _, row in self.lexicon_df.iterrows():
            skill_name = row['skill_name'].lower().strip()
            aliases = [skill_name]
            
            # Add aliases if present
            if pd.notna(row['aliases']):
                alias_list = str(row['aliases']).split(',')
                aliases.extend([a.strip().lower() for a in alias_list if a.strip()])
            
            # Use first alias as canonical
            canonical = skill_name
            skill_mapping[canonical] = list(set(aliases))
        
        return skill_mapping
    
    def _build_phrase_matcher(self) -> PhraseMatcher:
        """
        Build spaCy PhraseMatcher with all skill patterns.
        
        Returns:
            Configured PhraseMatcher
        """
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Add patterns for each skill
        for canonical, aliases in self.skill_mapping.items():
            patterns = [self.nlp.make_doc(alias) for alias in aliases]
            matcher.add(canonical, patterns)
        
        return matcher
    
    def _precompute_skill_embeddings(self):
        """Precompute embeddings for all canonical skill names."""
        skill_names = list(self.skill_mapping.keys())
        logger.info(f"Computing embeddings for {len(skill_names)} skills...")
        self.skill_embeddings = self.embedding_model.encode(
            skill_names,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        self.skill_names_list = skill_names
    
    def extract_with_phrases(self, text: str) -> Set[str]:
        """
        Extract skills using PhraseMatcher (exact matches).
        
        Args:
            text: Input text
            
        Returns:
            Set of extracted canonical skill names
        """
        if not text or pd.isna(text):
            return set()
        
        doc = self.nlp(text.lower())
        matches = self.phrase_matcher(doc)
        
        # Extract unique canonical skill names
        skills = set()
        for match_id, start, end in matches:
            canonical_name = self.nlp.vocab.strings[match_id]
            skills.add(canonical_name)
        
        return skills
    
    def extract_with_embeddings(self, text: str, top_k: int = 20) -> Set[str]:
        """
        Extract skills using embedding similarity (fuzzy matching).
        
        Args:
            text: Input text
            top_k: Maximum number of skills to consider
            
        Returns:
            Set of extracted canonical skill names
        """
        if not self.use_embeddings or not text or pd.isna(text):
            return set()
        
        # Encode text
        text_embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        
        # Compute similarities
        similarities = util.cos_sim(text_embedding, self.skill_embeddings)[0]
        
        # Get top-k matches above threshold
        top_results = similarities.topk(k=min(top_k, len(similarities)))
        
        skills = set()
        for score, idx in zip(top_results.values, top_results.indices):
            if score >= self.similarity_threshold:
                skills.add(self.skill_names_list[idx])
        
        return skills
    
    def extract_skills(self, text: str) -> Set[str]:
        """
        Extract skills using hybrid approach.
        
        Args:
            text: Input text (job description)
            
        Returns:
            Set of extracted canonical skill names
        """
        # Always use phrase matching
        phrase_skills = self.extract_with_phrases(text)
        
        # Optionally add embedding-based matches
        if self.use_embeddings:
            embedding_skills = self.extract_with_embeddings(text)
            return phrase_skills.union(embedding_skills)
        
        return phrase_skills
    
    def extract_from_dataframe(self, df: pd.DataFrame, 
                               text_column: str = 'description_clean') -> pd.DataFrame:
        """
        Extract skills from all rows in dataframe.
        
        Args:
            df: Input dataframe
            text_column: Column containing text to analyze
            
        Returns:
            Dataframe with added skill columns
        """
        logger.info(f"Extracting skills from {len(df)} job postings...")
        
        # Extract skills for each row
        skills_list = []
        for idx, row in df.iterrows():
            text = row.get(text_column, '')
            skills = self.extract_skills(text)
            skills_list.append(skills)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} rows")
        
        # Add extracted_skills column (semicolon-separated)
        df['extracted_skills'] = [';'.join(sorted(skills)) for skills in skills_list]
        df['num_skills'] = [len(skills) for skills in skills_list]
        
        # Create boolean columns for top N skills
        top_skills = self._get_top_skills(skills_list, top_n=50)
        logger.info(f"Creating boolean columns for top {len(top_skills)} skills")
        
        for skill in top_skills:
            column_name = f"has_{skill.replace(' ', '_').replace('-', '_').lower()}"
            df[column_name] = [skill in skills for skills in skills_list]
        
        logger.info("Skill extraction complete")
        
        return df
    
    def _get_top_skills(self, skills_list: List[Set[str]], top_n: int = 50) -> List[str]:
        """
        Get most frequently occurring skills.
        
        Args:
            skills_list: List of skill sets
            top_n: Number of top skills to return
            
        Returns:
            List of top skill names
        """
        skill_counts = {}
        for skills in skills_list:
            for skill in skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Sort by frequency
        sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        return [skill for skill, count in sorted_skills[:top_n]]
    
    def get_skill_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate skill extraction statistics.
        
        Args:
            df: Dataframe with extracted skills
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_jobs': len(df),
            'jobs_with_skills': (df['num_skills'] > 0).sum(),
            'avg_skills_per_job': df['num_skills'].mean(),
            'median_skills_per_job': df['num_skills'].median(),
            'max_skills_per_job': df['num_skills'].max(),
            'total_unique_skills': len(set(skill for skills in df['extracted_skills'].str.split(';') 
                                          for skill in skills if skill))
        }
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract skills from job postings using hybrid approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction (PhraseMatcher only)
  python scripts/skill_extractor.py --input data/cleaned_jobs.csv --output data/jobs_with_skills.csv
  
  # With embedding-based fallback
  python scripts/skill_extractor.py --input data/cleaned_jobs.csv --output data/jobs_with_skills.csv --use-embeddings
  
  # Custom similarity threshold
  python scripts/skill_extractor.py --input data/cleaned_jobs.csv --output data/jobs_with_skills.csv --use-embeddings --similarity-threshold 0.80
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file (preprocessed jobs)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file (with extracted skills)')
    parser.add_argument('--lexicon', type=str, default='skills/skills_lexicon.csv',
                       help='Path to skills lexicon CSV')
    parser.add_argument('--text-column', type=str, default='description_clean',
                       help='Column containing text to analyze')
    parser.add_argument('--use-embeddings', action='store_true',
                       help='Enable embedding-based fuzzy matching')
    parser.add_argument('--no-embeddings', dest='use_embeddings', action='store_false',
                       help='Disable embedding-based matching (faster)')
    parser.add_argument('--similarity-threshold', type=float, default=0.78,
                       help='Cosine similarity threshold for embeddings (0.0-1.0)')
    parser.add_argument('--embedding-model', type=str, default='all-MiniLM-L6-v2',
                       help='Sentence-BERT model name')
    parser.add_argument('--chunksize', type=int,
                       help='Process large files in chunks')
    
    parser.set_defaults(use_embeddings=False)
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    if not Path(args.lexicon).exists():
        logger.error(f"Lexicon file not found: {args.lexicon}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize extractor
        extractor = SkillExtractor(
            lexicon_path=args.lexicon,
            use_embeddings=args.use_embeddings,
            similarity_threshold=args.similarity_threshold,
            model_name=args.embedding_model
        )
        
        # Load and process data
        logger.info(f"Loading data from {args.input}")
        
        if args.chunksize:
            # Process in chunks
            logger.info(f"Processing in chunks of {args.chunksize}")
            chunks = []
            for chunk in pd.read_csv(args.input, chunksize=args.chunksize):
                processed_chunk = extractor.extract_from_dataframe(chunk, args.text_column)
                chunks.append(processed_chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(args.input)
            df = extractor.extract_from_dataframe(df, args.text_column)
        
        # Save output
        logger.info(f"Saving results to {args.output}")
        df.to_csv(args.output, index=False)
        
        # Print statistics
        stats = extractor.get_skill_statistics(df)
        
        logger.info("=" * 60)
        logger.info("SKILL EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total jobs processed: {stats['total_jobs']}")
        logger.info(f"Jobs with skills: {stats['jobs_with_skills']} ({stats['jobs_with_skills']/stats['total_jobs']*100:.1f}%)")
        logger.info(f"Average skills per job: {stats['avg_skills_per_job']:.2f}")
        logger.info(f"Median skills per job: {stats['median_skills_per_job']:.0f}")
        logger.info(f"Max skills per job: {stats['max_skills_per_job']}")
        logger.info(f"Total unique skills found: {stats['total_unique_skills']}")
        logger.info(f"Embedding-based matching: {'Enabled' if args.use_embeddings else 'Disabled'}")
        if args.use_embeddings:
            logger.info(f"Similarity threshold: {args.similarity_threshold}")
        logger.info("=" * 60)
        logger.info("Skill extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Skill extraction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()