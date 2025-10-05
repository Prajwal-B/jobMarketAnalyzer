@echo off
REM file: run_all.bat
REM End-to-end pipeline execution script for Windows

echo ==========================================
echo Job Market Analyzer - Full Pipeline
echo ==========================================
echo.

REM Step 1: Create directories
echo [1/8] Creating directories...
if not exist data mkdir data
if not exist exports\tableau_ready mkdir exports\tableau_ready
if not exist exports\by_region mkdir exports\by_region
if not exist logs mkdir logs

REM Step 2: Check Python version
echo [2/8] Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Step 3: Create virtual environment (optional)
if not exist venv (
    echo [3/8] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Virtual environment created and activated
) else (
    echo [3/8] Activating existing virtual environment...
    call venv\Scripts\activate.bat
)

REM Step 4: Install dependencies
echo [4/8] Installing dependencies...
python -m pip install --upgrade pip > nul 2>&1
pip install -r requirements.txt > nul 2>&1
python -m spacy download en_core_web_sm > nul 2>&1
echo Dependencies installed

REM Step 5: Data collection
echo [5/8] Running data collection (sample mode)...
python scripts\data_collection.py --use-sample --output data\raw_jobs.csv > logs\data_collection.log 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Data collection failed. Check logs\data_collection.log
    pause
    exit /b 1
)
echo Data collection complete

REM Step 6: Preprocessing
echo [6/8] Running preprocessing...
python scripts\preprocess.py --input data\raw_jobs.csv --output data\cleaned_jobs.csv > logs\preprocess.log 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Preprocessing failed. Check logs\preprocess.log
    pause
    exit /b 1
)
echo Preprocessing complete

REM Step 7: Skill extraction
echo [7/8] Running skill extraction...
python scripts\skill_extractor.py --input data\cleaned_jobs.csv --output data\jobs_with_skills.csv --lexicon skills\skills_lexicon.csv > logs\skill_extraction.log 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Skill extraction failed. Check logs\skill_extraction.log
    pause
    exit /b 1
)
echo Skill extraction complete

REM Step 8: Analysis pipeline
echo [8/8] Running analysis pipeline...

echo   - Co-occurrence analysis...
python scripts\skill_cooccurrence.py --input data\jobs_with_skills.csv --output exports\cooccurrence.csv > logs\cooccurrence.log 2>&1

echo   - Trend analysis...
python scripts\trend_analysis.py --input data\jobs_with_skills.csv --output exports\trends.csv > logs\trends.log 2>&1

echo   - Geographic analysis...
python scripts\geography_analysis.py --input data\jobs_with_skills.csv --output-dir exports\by_region\ > logs\geography.log 2>&1

echo Analysis complete
echo.

REM Summary
echo ==========================================
echo Pipeline execution completed!
echo ==========================================
echo.
echo Output files:
echo   - data\cleaned_jobs.csv          (cleaned job postings)
echo   - data\jobs_with_skills.csv      (with extracted skills)
echo   - exports\cooccurrence.csv       (skill co-occurrence matrix)
echo   - exports\skill_network.graphml  (network graph)
echo   - exports\trends.csv             (time series analysis)
echo   - exports\by_region\*.csv        (regional breakdowns)
echo   - exports\tableau_ready\*.csv    (Tableau imports)
echo.
echo Logs saved in logs\ directory
echo.
echo Next steps:
echo   - Review outputs in exports\ directory
echo   - Open notebooks\analysis_demo.ipynb for visualizations
echo   - Import exports\tableau_ready\*.csv into Tableau
echo   - Run 'pytest tests\' to verify installation
echo.
pause