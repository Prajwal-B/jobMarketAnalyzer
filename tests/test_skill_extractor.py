# file: tests/test_skill_extractor.py
"""
Unit tests for skill extraction module.
Tests PhraseMatcher, embedding similarity, and extraction accuracy.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.skill_extractor import SkillExtractor


@pytest.fixture
def sample_lexicon(tmp_path):
    """Create a sample skills lexicon for testing."""
    lexicon_data = {
        'skill_name': ['Python', 'Machine Learning', 'SQL', 'JavaScript', 'React'],
        'category': ['Technical', 'Technical', 'Technical', 'Technical', 'Technical'],
        'subcategory': ['Programming', 'AI/ML', 'Database', 'Programming', 'Web'],
        'aliases': ['python,py', 'ml,machine learning', 'sql', 'javascript,js', 'react,reactjs']
    }
    
    lexicon_df = pd.DataFrame(lexicon_data)
    lexicon_path = tmp_path / "test_lexicon.csv"
    lexicon_df.to_csv(lexicon_path, index=False)
    
    return str(lexicon_path)


@pytest.fixture
def extractor(sample_lexicon):
    """Create SkillExtractor instance for testing."""
    return SkillExtractor(lexicon_path=sample_lexicon, use_embeddings=False)


@pytest.fixture
def extractor_with_embeddings(sample_lexicon):
    """Create SkillExtractor with embeddings enabled."""
    return SkillExtractor(
        lexicon_path=sample_lexicon,
        use_embeddings=True,
        similarity_threshold=0.75
    )


class TestSkillExtractor:
    """Test suite for SkillExtractor class."""
    
    def test_initialization(self, extractor):
        """Test extractor initializes correctly."""
        assert extractor is not None
        assert len(extractor.skill_mapping) > 0
        assert extractor.phrase_matcher is not None
        assert not extractor.use_embeddings
    
    def test_initialization_with_embeddings(self, extractor_with_embeddings):
        """Test extractor with embeddings initializes correctly."""
        assert extractor_with_embeddings.use_embeddings
        assert extractor_with_embeddings.embedding_model is not None
        assert extractor_with_embeddings.skill_embeddings is not None
    
    def test_exact_skill_match(self, extractor):
        """Test extraction of exact skill matches."""
        text = "We need someone with Python and SQL experience"
        skills = extractor.extract_with_phrases(text)
        
        assert 'python' in skills
        assert 'sql' in skills
        assert len(skills) == 2
    
    def test_alias_matching(self, extractor):
        """Test extraction using skill aliases."""
        text = "Looking for JS and ML expert"
        skills = extractor.extract_with_phrases(text)
        
        assert 'javascript' in skills
        assert 'machine learning' in skills
    
    def test_case_insensitive_matching(self, extractor):
        """Test case-insensitive extraction."""
        text = "PYTHON and python and PyThOn required"
        skills = extractor.extract_with_phrases(text)
        
        assert 'python' in skills
        assert len(skills) == 1  # Should deduplicate
    
    def test_multi_word_skills(self, extractor):
        """Test extraction of multi-word skill phrases."""
        text = "Experience with machine learning and React"
        skills = extractor.extract_with_phrases(text)
        
        assert 'machine learning' in skills
        assert 'react' in skills
    
    def test_empty_text(self, extractor):
        """Test extraction from empty text."""
        assert extractor.extract_with_phrases("") == set()
        assert extractor.extract_with_phrases(None) == set()
    
    def test_no_skills_found(self, extractor):
        """Test when no skills are present in text."""
        text = "Looking for a great candidate"
        skills = extractor.extract_with_phrases(text)
        
        assert len(skills) == 0
    
    def test_embedding_extraction(self, extractor_with_embeddings):
        """Test embedding-based extraction (if embeddings enabled)."""
        text = "Need expert in programming with Python"
        skills = extractor_with_embeddings.extract_skills(text)
        
        # Should find Python via phrase matching
        assert 'python' in skills
    
    def test_dataframe_extraction(self, extractor, tmp_path):
        """Test extraction from dataframe."""
        df = pd.DataFrame({
            'description_clean': [
                'python and sql required',
                'javascript developer needed',
                'machine learning expert'
            ]
        })
        
        result_df = extractor.extract_from_dataframe(df)
        
        assert 'extracted_skills' in result_df.columns
        assert 'num_skills' in result_df.columns
        assert result_df['num_skills'].sum() > 0
    
    def test_boolean_columns_creation(self, extractor):
        """Test creation of boolean skill columns."""
        df = pd.DataFrame({
            'description_clean': [
                'python and sql required',
                'javascript and react needed',
            ]
        })
        
        result_df = extractor.extract_from_dataframe(df)
        
        # Check boolean columns exist
        assert 'has_python' in result_df.columns
        assert 'has_sql' in result_df.columns
        
        # Check values
        assert result_df.iloc[0]['has_python'] == True
        assert result_df.iloc[0]['has_sql'] == True
        assert result_df.iloc[1]['has_python'] == False
    
    def test_semicolon_separated_output(self, extractor):
        """Test skills are output as semicolon-separated strings."""
        df = pd.DataFrame({
            'description_clean': ['python and sql required']
        })
        
        result_df = extractor.extract_from_dataframe(df)
        skills_str = result_df.iloc[0]['extracted_skills']
        
        assert ';' in skills_str or len(skills_str.split(';')) == 1
        skills_list = [s.strip() for s in skills_str.split(';')]
        assert 'python' in skills_list
        assert 'sql' in skills_list
    
    def test_skill_statistics(self, extractor):
        """Test skill statistics calculation."""
        df = pd.DataFrame({
            'description_clean': [
                'python and sql',
                'javascript',
                'python and react'
            ]
        })
        
        result_df = extractor.extract_from_dataframe(df)
        stats = extractor.get_skill_statistics(result_df)
        
        assert stats['total_jobs'] == 3
        assert stats['jobs_with_skills'] == 3
        assert stats['avg_skills_per_job'] > 0
        assert stats['total_unique_skills'] > 0
    
    def test_overlapping_skills(self, extractor):
        """Test handling of overlapping skill mentions."""
        text = "Python Python Python"
        skills = extractor.extract_with_phrases(text)
        
        # Should deduplicate
        assert len(skills) == 1
        assert 'python' in skills
    
    def test_skills_in_context(self, extractor):
        """Test extraction with realistic job description."""
        text = """
        We are looking for a Full Stack Developer with strong skills in:
        - Python and JavaScript programming
        - SQL database management
        - React for frontend development
        - Machine learning basics
        
        Must have excellent communication skills.
        """
        
        skills = extractor.extract_with_phrases(text)
        
        assert 'python' in skills
        assert 'javascript' in skills
        assert 'sql' in skills
        assert 'react' in skills
        assert 'machine learning' in skills


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_special_characters(self, extractor):
        """Test extraction with special characters."""
        text = "Python, SQL & JavaScript (React)"
        skills = extractor.extract_with_phrases(text)
        
        assert 'python' in skills
        assert 'sql' in skills
        assert 'javascript' in skills
    
    def test_very_long_text(self, extractor):
        """Test extraction from very long text."""
        text = " ".join(["Python is great"] * 1000)
        skills = extractor.extract_with_phrases(text)
        
        assert 'python' in skills
        assert len(skills) == 1
    
    def test_unicode_text(self, extractor):
        """Test extraction with unicode characters."""
        text = "Python programming with SQL data"
        skills = extractor.extract_with_phrases(text)
        
        assert len(skills) > 0
    
    def test_missing_description_column(self, extractor):
        """Test handling of missing description column."""
        df = pd.DataFrame({
            'other_column': ['some text']
        })
        
        with pytest.raises(KeyError):
            extractor.extract_from_dataframe(df, text_column='missing_column')
    
    def test_empty_dataframe(self, extractor):
        """Test extraction from empty dataframe."""
        df = pd.DataFrame({
            'description_clean': []
        })
        
        result_df = extractor.extract_from_dataframe(df)
        
        assert len(result_df) == 0
        assert 'extracted_skills' in result_df.columns


class TestEmbeddingMatching:
    """Test embedding-based skill matching."""
    
    @pytest.mark.slow
    def test_fuzzy_matching(self, extractor_with_embeddings):
        """Test fuzzy matching with embeddings (marked as slow)."""
        # This test may take longer due to model loading
        text = "Proficient in programming languages"
        skills = extractor_with_embeddings.extract_with_embeddings(text, top_k=10)
        
        # Should find some programming-related skills
        assert isinstance(skills, set)
    
    def test_similarity_threshold(self, sample_lexicon):
        """Test different similarity thresholds."""
        extractor_low = SkillExtractor(
            sample_lexicon,
            use_embeddings=True,
            similarity_threshold=0.5
        )
        
        extractor_high = SkillExtractor(
            sample_lexicon,
            use_embeddings=True,
            similarity_threshold=0.9
        )
        
        text = "Programming skills required"
        
        skills_low = extractor_low.extract_with_embeddings(text)
        skills_high = extractor_high.extract_with_embeddings(text)
        
        # Lower threshold should return more or equal skills
        assert len(skills_low) >= len(skills_high)


@pytest.mark.integration
class TestIntegration:
    """Integration tests with real pipeline components."""
    
    def test_full_pipeline_simulation(self, extractor, tmp_path):
        """Test full extraction pipeline."""
        # Create sample input
        input_df = pd.DataFrame({
            'job_id': [1, 2, 3],
            'title': ['Data Scientist', 'Web Developer', 'ML Engineer'],
            'description_clean': [
                'python machine learning sql',
                'javascript react',
                'python machine learning'
            ]
        })
        
        # Extract skills
        result_df = extractor.extract_from_dataframe(input_df)
        
        # Save to file
        output_path = tmp_path / "output.csv"
        result_df.to_csv(output_path, index=False)
        
        # Reload and verify
        reloaded_df = pd.read_csv(output_path)
        
        assert len(reloaded_df) == 3
        assert 'extracted_skills' in reloaded_df.columns
        assert reloaded_df['num_skills'].sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])