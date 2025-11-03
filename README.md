# Job Market Analyzer & Skill Extractor

**Complete modular pipeline for analyzing India's job market**

## Overview

This project provides an end-to-end data engineering and NLP pipeline to:
- Ingest job postings from CSV datasets (Kaggle) or web scraping templates
- Preprocess and normalize job titles, locations, and descriptions
- Extract technical and soft skills using hybrid approaches (rule-based + embeddings)
- Analyze skill co-occurrence, trends over time, and geographic distributions
- Export analysis-ready datasets for Tableau visualization

## Features

- **Hybrid Skill Extraction**: PhraseMatcher (spaCy) + Sentence-BERT embeddings
- **Time Series Analysis**: Monthly/quarterly trends with anomaly detection
- **Network Analysis**: Skill co-occurrence graphs (NetworkX)
- **Geographic Mapping**: Tier-1/Tier-2 city classification
- **Tableau-Ready Exports**: Pre-formatted CSVs with boolean skill columns
- **Optional NER Training**: Skeleton for fine-tuning custom skill recognizers
- **Fully Local**: No external API keys required
- **Docker Support**: Reproducible containerized environment

## Quick Start

### 1. Setup
```bash
# Clone/extract the project
cd job-market-analyzer

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
