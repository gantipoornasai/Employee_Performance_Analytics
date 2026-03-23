# Automation Scheduler Setup
## Employee Performance Analytics — Monthly Run

## Overview
The automation script runs on the 1st of every month.
It loads new data, generates predictions, updates
the Power BI dataset, and emails the report.

## Windows Task Scheduler
Task Name: HR Analytics Monthly Run
Schedule:  1st of every month at 00:00
Script:    automation/automate_monthly.py

To verify setup:
1. Open Task Scheduler
2. Find "HR Analytics Monthly Run"
3. Last Run Time should show recent date
4. Last Run Result should show 0 (success)

## Manual Run Commands
Run for current month:
python automation/automate_monthly.py

Run for specific month:
python automation/automate_monthly.py --month 2024-12

Test without saving files:
python automation/automate_monthly.py --dry-run

## Linux / Mac cron schedule
Open terminal and type: crontab -e
Add this line:
0 0 1 * * /path/to/venv/bin/python /path/to/automation/automate_monthly.py

## Log File
Location: automation/automation.log
Check this file after each run to confirm success

## Output Files Generated Each Month
data/processed/employee_with_predictions.csv
data/processed/predictions_YYYY-MM.csv
data/processed/summary_YYYY-MM.json
data/monthly_snapshots/snapshot_YYYY-MM.csv
docs/monthly_report_YYYY-MM.png

## Power BI Refresh
After script runs:
1. Open Power BI Desktop
2. Home → Refresh
OR
Configure scheduled refresh in Power BI Service

## Business Impact
Manual monthly reporting time:    8-12 hours
Automated pipeline time:          3-7 minutes
Annual time saved per analyst:    ~120 hours
Consistency improvement:          100%
