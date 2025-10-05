#!/bin/bash
# file: run_all.sh
# End-to-end pipeline execution script for Unix/Linux/Mac

set -e  # Exit on error

echo "=========================================="
echo "Job Market Analyzer - Full Pipeline"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Create directories
echo -e "${BLUE}[1/8] Creating directories...${NC}"
mkdir -p data exports/tableau_ready exports/by_region logs

# Step 2: Check Python version
echo -e "${BLUE}[2/8] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Step 3: Create virtual environment (optional)
if [ ! -d "venv" ]; then
    echo -e "${BLUE}[3/8] Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}Virtual environment created and activated${NC}"
else
    echo -e "${BLUE}[3/8] Activating existing virtual environment...${NC}"
    source venv/bin/activate
fi

# Step 4: Install dependencies
echo -e "${BLUE}[4/8] Installing dependencies...${NC}"
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
python -m spacy download en_core_web_sm > /dev/null 2>&1
echo -e "${GREEN}Dependencies installed${NC}"

# Step 5: Data collection
echo -e "${BLUE}[5/8] Running data collection (sample mode)...${NC}"
python scripts/data_collection.py \
    --use-sample \
    --output data/raw_jobs.csv \
    2>&1 | tee logs/data_collection.log
echo -e "${GREEN}Data collection complete${NC}"

# Step 6: Preprocessing
echo -e "${BLUE}[6/8] Running preprocessing...${NC}"
python scripts/preprocess.py \
    --input data/raw_jobs.csv \
    --output data/cleaned_jobs.csv \
    2>&1 | tee logs/preprocess.log
echo -e "${GREEN}Preprocessing complete${NC}"

# Step 7: Skill extraction
echo -e "${BLUE}[7/8] Running skill extraction...${NC}"
python scripts/skill_extractor.py \
    --input data/cleaned_jobs.csv \
    --output data/jobs_with_skills.csv \
    --lexicon skills/skills_lexicon.csv \
    2>&1 | tee logs/skill_extraction.log
echo -e "${GREEN}Skill extraction complete${NC}"

# Step 8: Analysis pipeline
echo -e "${BLUE}[8/8] Running analysis pipeline...${NC}"

echo "  - Co-occurrence analysis..."
python scripts/skill_cooccurrence.py \
    --input data/jobs_with_skills.csv \
    --output exports/cooccurrence.csv \
    2>&1 | tee logs/cooccurrence.log

echo "  - Trend analysis..."
python scripts/trend_analysis.py \
    --input data/jobs_with_skills.csv \
    --output exports/trends.csv \
    2>&1 | tee logs/trends.log

echo "  - Geographic analysis..."
python scripts/geography_analysis.py \
    --input data/jobs_with_skills.csv \
    --output-dir exports/by_region/ \
    2>&1 | tee logs/geography.log

echo -e "${GREEN}Analysis complete${NC}"

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}Pipeline execution completed!${NC}"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - data/cleaned_jobs.csv          (cleaned job postings)"
echo "  - data/jobs_with_skills.csv      (with extracted skills)"
echo "  - exports/cooccurrence.csv       (skill co-occurrence matrix)"
echo "  - exports/skill_network.graphml  (network graph)"
echo "  - exports/trends.csv             (time series analysis)"
echo "  - exports/by_region/*.csv        (regional breakdowns)"
echo "  - exports/tableau_ready/*.csv    (Tableau imports)"
echo ""
echo "Logs saved in logs/ directory"
echo ""
echo "Next steps:"
echo "  - Review outputs in exports/ directory"
echo "  - Open notebooks/analysis_demo.ipynb for visualizations"
echo "  - Import exports/tableau_ready/*.csv into Tableau"
echo "  - Run 'pytest tests/' to verify installation"
echo ""