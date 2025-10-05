# file: exports/tableau_ready/README.md

# Tableau Integration Guide

This directory contains CSV exports optimized for Tableau visualization.

## Available Files

After running the pipeline, you'll find the following files here:

1. **jobs_tableau.csv** - Main dataset with all job postings
2. **jobs_geographic.csv** - Geographic-focused export with tier/region data
3. **skill_trends.csv** - Time series data for trend analysis
4. **skill_network.csv** - Network edge list for relationship mapping

## Quick Start with Tableau

### Option 1: Tableau Desktop

1. Open Tableau Desktop
2. Click "Connect" → "Text file"
3. Navigate to this directory and select `jobs_tableau.csv`
4. Click "Update Now" to load data
5. Begin creating visualizations

### Option 2: Tableau Public

1. Open Tableau Public
2. Click "Connect to Data" → "Text file"
3. Select the CSV file
4. Publish to Tableau Public (free)

## Data Schema

### jobs_tableau.csv

Main export with all job information and extracted skills.

**Dimensions:**
- `job_id` - Unique identifier
- `title` - Normalized job title
- `company` - Company name
- `location` - Normalized city name
- `city_tier` - Tier-1, Tier-2, or Other
- `region` - North, South, East, West, Central
- `posted_date` - Original posting date
- `job_type` - Full-time, Part-time, Contract, etc.

**Dates:**
- `year` - Posted year (for filtering)
- `month` - Posted month (1-12)
- `posted_quarter` - Q1, Q2, Q3, Q4

**Measures:**
- `salary_min` - Minimum salary (INR)
- `salary_max` - Maximum salary (INR)
- `experience_years` - Required experience
- `num_skills` - Count of extracted skills

**Boolean Skill Columns:**
- `has_python` - TRUE if Python skill present
- `has_sql` - TRUE if SQL skill present
- `has_machine_learning` - TRUE if ML skill present
- ... (50+ boolean columns for top skills)

### jobs_geographic.csv

Focused on geographic analysis with aggregations.

**Additional Fields:**
- Pre-calculated skill counts by region
- Average salaries by tier
- Regional skill distribution

### skill_trends.csv

Time series data for trend visualization.

**Columns:**
- `period` - Time period (YYYY-MM or YYYY-QX)
- `skill` - Skill name
- `count` - Number of mentions
- `percentage` - % of jobs in period

## Recommended Visualizations

### 1. Geographic Heat Map

**Type:** Map
**Dimensions:** Location (Latitude/Longitude)
**Measures:** COUNT(job_id)
**Color:** City Tier
**Tooltip:** Location, Count, Top 3 Skills

### 2. Skill Demand Dashboard

**Sheets:**
- Bar chart: Top 20 skills by count
- Tree map: Skills colored by category
- Line chart: Trend over time
- Filters: Location, Date Range, Experience Level

### 3. Salary Analysis

**Type:** Box Plot
**Rows:** Location (filtered to top 10)
**Columns:** AVG(salary_min + salary_max) / 2
**Color:** City Tier
**Filters:** Experience range, Job title

### 4. Trend Analysis

**Type:** Line Chart
**Columns:** Month/Quarter (continuous)
**Rows:** Skill Count
**Color:** Skill (filtered to top 10)
**Filters:** Date range, Region

### 5. Skill Co-occurrence Matrix

**Type:** Heatmap
**Rows:** Skill 1
**Columns:** Skill 2
**Color:** Co-occurrence count
**Filters:** Minimum co-occurrence threshold

### 6. Regional Comparison

**Type:** Stacked Bar Chart
**Rows:** Region
**Columns:** COUNT(job_id)
**Color:** Top 10 Skills (stacked)
**Filters:** Date range

## Creating Your First Dashboard

### Step-by-Step Example: "Top Skills by City"

1. **Connect Data:**
   - Load `jobs_tableau.csv`

2. **Create Sheet 1 - Map:**
   - Drag `Location` to Detail
   - Tableau will auto-generate Latitude/Longitude
   - Drag `job_id` to Size (use COUNT)
   - Drag `City Tier` to Color
   - Add `num_skills` to Tooltip (AVG)

3. **Create Sheet 2 - Top Skills Bar Chart:**
   - Create calculated field: `Has Skill` =
IF [has_python] THEN "Python"
 ELSEIF [has_sql] THEN "SQL"
 ... (repeat for top skills)
 END
- Drag `Has Skill` to Rows
   - Drag `job_id` to Columns (COUNT)
   - Sort descending
   - Limit to top 20

4. **Create Sheet 3 - Timeline:**
   - Drag `posted_date` to Columns (convert to Month)
   - Drag `job_id` to Rows (COUNT)
   - Add trend line

5. **Combine in Dashboard:**
   - New Dashboard
   - Drag all three sheets
   - Add filters: Date Range, City Tier, Region
   - Set filter actions: clicking map filters other sheets

## Advanced Techniques

### Calculated Fields

**Average Salary:**
([salary_min] + [salary_max]) / 2

**Salary Category:**
IF [Average Salary] < 500000 THEN "Entry (<5L)"
ELSEIF [Average Salary] < 1000000 THEN "Mid (5-10L)"
ELSEIF [Average Salary] < 1500000 THEN "Senior (10-15L)"
ELSE "Expert (>15L)"
END

**Skill Diversity Score:**
[num_skills] / [experience_years]

**Growth Rate (requires parameters):**
(SUM(IIF([posted_date] >= [End Period], [job_id], 0)) -
SUM(IIF([posted_date] < [Start Period], [job_id], 0))) /
SUM(IIF([posted_date] < [Start Period], [job_id], 0))

### Parameters for Interactivity

1. **Top N Skills Parameter:**
   - Create integer parameter: 5, 10, 20, 50
   - Use in calculated field to filter top N skills

2. **Date Range Parameters:**
   - Start Date and End Date parameters
   - Filter data based on user selection

3. **Salary Range Slider:**
   - Min/Max salary parameters
   - Dynamic filtering

### Level of Detail (LOD) Expressions

**Skills per Company:**
{FIXED [company] : AVG([num_skills])}

**Most Common Skill in Location:**
{FIXED [location] : MAX([skill_count])}

## Performance Tips

1. **Extract vs. Live Connection:**
   - For <100K rows: Live connection works fine
   - For >100K rows: Create data extract (.hyper file)

2. **Optimize Boolean Columns:**
   - Use only top 30-50 skill columns
   - Remove rare skills to reduce data size

3. **Aggregation:**
   - Pre-aggregate in Python if possible
   - Use Tableau's aggregate data sources

4. **Filters:**
   - Use context filters for large datasets
   - Apply filters to data source, not worksheet

## Sharing & Publishing

### Tableau Public (Free)

1. Create visualization
2. File → Save to Tableau Public
3. Share generated URL

**Note:** Data becomes public. Remove sensitive information.

### Tableau Server/Online (Paid)

1. Publish workbook to server
2. Set permissions and refresh schedules
3. Embed in websites using JavaScript API

## Troubleshooting

**Issue:** Location not mapping correctly
**Solution:** Ensure city names match Tableau's geographic database or add custom geocoding

**Issue:** Boolean columns showing as measures
**Solution:** Convert to dimensions by dragging to Dimensions pane

**Issue:** Date not parsing correctly
**Solution:** Convert text to date: Right-click → Change Data Type → Date

**Issue:** Too many marks (>100K warning)
**Solution:** Aggregate data, filter date range, or create data extract

## Example Dashboards

Sample dashboard specifications included in `sample_export.csv`.

**Dashboard 1: Executive Summary**
- KPIs: Total jobs, unique skills, avg salary
- Map: Jobs by location
- Bar chart: Top 10 skills
- Trend line: Jobs over time

**Dashboard 2: Skill Deep Dive**
- Skill selector parameter
- Time series for selected skill
- Geographic distribution
- Co-occurring skills network
- Salary range for jobs with skill

**Dashboard 3: Regional Analysis**
- Map with regional overlay
- Skill distribution by region
- Salary comparison by tier
- Top companies by region

## Resources

- **Tableau Public Gallery:** https://public.tableau.com/gallery
- **Training:** https://www.tableau.com/learn/training
- **Community Forums:** https://community.tableau.com/
- **Sample Workbooks:** Download from Tableau Public, search "job market analysis"

## Support

For issues with data exports, see main project README or check logs in `logs/` directory.

For Tableau-specific help, consult Tableau documentation or community forums.

---

**Quick Links:**
- [Tableau Public Download](https://public.tableau.com/en-us/s/download)
- [Tableau Desktop Trial](https://www.tableau.com/products/trial)
- [Data Visualization Best Practices](https://help.tableau.com/current/blueprint/en-us/bp_visual_best_practices.htm)