# automate_monthly.py
# ================================================
# Monthly Performance Analytics Automation Script
#
# PURPOSE:
#     Runs on the 1st of every month automatically
#     Simulates a production HR analytics pipeline
#     Load > Process > Predict > Export > Report
#
# USAGE:
#     python automation\automate_monthly.py
#     python automation\automate_monthly.py --month 2024-12
#     python automation\automate_monthly.py --dry-run
#
# SCHEDULE:
#     Windows: Task Scheduler (see scheduler_setup.md)
#     Linux/Mac: cron  0 0 1 * *
# ================================================

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

# ---- Setup project paths ----------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# ---- Setup logging ----------------------------------
log_path = SCRIPT_DIR / 'automation.log'

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s  [%(levelname)s]  %(message)s',
    datefmt  = '%Y-%m-%d %H:%M:%S',
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            log_path,
            encoding='utf-8'
        )
    ]
)
log = logging.getLogger(__name__)


# =====================================================
# STEP 1 - PARSE ARGUMENTS
# =====================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Monthly HR Analytics Automation'
    )
    parser.add_argument(
        '--month',
        type    = str,
        default = datetime.now().strftime('%Y-%m'),
        help    = 'Month to process in YYYY-MM format'
    )
    parser.add_argument(
        '--dry-run',
        action  = 'store_true',
        help    = 'Run without saving any files'
    )
    return parser.parse_args()


# =====================================================
# STEP 2 - LOAD MONTHLY DATA
# =====================================================
def load_monthly_data(month_str):
    """
    In production: connects to HRIS API or data lake.
    In demo: simulates new month by adding noise.
    """
    log.info(f"Loading data for month: {month_str}")

    snapshot_dir  = PROJECT_ROOT / 'data' / 'monthly_snapshots'
    snapshot_path = snapshot_dir / f'snapshot_{month_str}.csv'
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if snapshot_path.exists():
        log.info(f"Loading existing snapshot: {snapshot_path}")
        df = pd.read_csv(snapshot_path)

    else:
        log.info("No snapshot found - simulating monthly update")

        # Try to load feature-engineered data first
        base_path = (
            PROJECT_ROOT / 'data' / 'processed' /
            'employee_features.csv'
        )
        if not base_path.exists():
            base_path = (
                PROJECT_ROOT / 'data' / 'raw' /
                'employee_performance_raw.csv'
            )

        if not base_path.exists():
            log.error(f"No base data found at: {base_path}")
            raise FileNotFoundError(
                "Base data file not found. "
                "Run Phase 3 notebook first."
            )

        df = pd.read_csv(base_path)
        log.info(f"Base data loaded: {base_path.name}")

        # Simulate monthly changes using reproducible seed
        seed = sum(ord(c) for c in month_str)
        rng  = np.random.RandomState(seed)

        if 'PerformanceRating' in df.columns:
            df['PerformanceRating'] = np.clip(
                df['PerformanceRating']
                + rng.normal(0, 0.08, len(df)),
                1, 5
            ).round(1)

        if 'OKRCompletionPct' in df.columns:
            df['OKRCompletionPct'] = np.clip(
                df['OKRCompletionPct']
                + rng.normal(0, 1.5, len(df)),
                0, 100
            ).round(1)

        if 'EngagementScore' in df.columns:
            df['EngagementScore'] = np.clip(
                df['EngagementScore']
                + rng.normal(0, 2.0, len(df)),
                0, 100
            ).round(1)

        if 'AbsenteeismDays' in df.columns:
            df['AbsenteeismDays'] = np.clip(
                df['AbsenteeismDays']
                + rng.randint(-1, 2, len(df)),
                0, 30
            )

        df['ReportMonth'] = month_str

        # Save snapshot
        df.to_csv(snapshot_path, index=False)
        log.info(f"Snapshot saved: {snapshot_path}")

    log.info(
        f"Data loaded: {len(df):,} employees, "
        f"{df.shape[1]} columns"
    )
    return df


# =====================================================
# STEP 3 - PREPROCESS DATA
# =====================================================
def preprocess_data(df):
    """
    Load saved sklearn pipeline and transform data.
    Same pipeline every month = consistent results.
    """
    log.info("Running preprocessing pipeline")

    pipeline_path = (
        PROJECT_ROOT / 'models' /
        'preprocessing_pipeline.joblib'
    )

    if not pipeline_path.exists():
        log.error(f"Pipeline not found: {pipeline_path}")
        raise FileNotFoundError(
            "Preprocessing pipeline not found. "
            "Run Phase 4 notebook first."
        )

    pipeline = joblib.load(pipeline_path)
    log.info("Pipeline loaded successfully")

    # Load feature column list
    feat_path = PROJECT_ROOT / 'models' / 'feature_cols.json'

    if feat_path.exists():
        with open(feat_path) as f:
            feature_cols = json.load(f)
        feature_cols = [
            c for c in feature_cols
            if c in df.columns
        ]
        log.info(f"Feature columns loaded: {len(feature_cols)}")
    else:
        log.warning("feature_cols.json not found - using fallback")
        exclude = [
            'PerformanceRating', 'HighPerformer',
            'EmployeeID', 'ManagerID', 'ReportMonth',
            'PercentSalaryHikeLast', 'BonusPayoutPct'
        ]
        feature_cols = [
            c for c in df.select_dtypes(
                include='number'
            ).columns
            if c not in exclude
        ]

    log.info(f"Features used for prediction: {len(feature_cols)}")

    X = df[feature_cols].copy()

    try:
        X_processed = pipeline.transform(X)
        log.info(
            f"Preprocessing complete - "
            f"shape: {X_processed.shape}"
        )
    except Exception as e:
        log.warning(f"Transform failed: {e}")
        log.info("Using fit_transform as fallback")
        X_processed = pipeline.fit_transform(X)

    return df, X_processed, feature_cols


# =====================================================
# STEP 4 - GENERATE PREDICTIONS
# =====================================================
def generate_predictions(df, X_processed):
    """
    Load saved XGBoost model and score all employees.
    """
    log.info("Generating ML predictions")

    model_path = (
        PROJECT_ROOT / 'models' /
        'xgb_performance_model.joblib'
    )

    if not model_path.exists():
        log.error(f"Model not found: {model_path}")
        raise FileNotFoundError(
            "XGBoost model not found. "
            "Run Phase 7 notebook first."
        )

    model = joblib.load(model_path)
    log.info("Model loaded successfully")

    proba = model.predict_proba(X_processed)[:, 1]
    preds = (proba >= 0.45).astype(int)

    df = df.copy()
    df['HighPerformerProbability'] = proba.round(4)
    df['PredictedHighPerformer']   = preds

    df['PredictionConfidence'] = pd.cut(
        df['HighPerformerProbability'],
        bins   = [0, 0.30, 0.50, 0.70, 0.85, 1.01],
        labels = [
            'Very Low', 'Low',
            'Moderate', 'High', 'Very High'
        ]
    ).astype(str)

    hp_count = int(preds.sum())
    log.info(
        f"Predictions done: {hp_count:,} high performers "
        f"({hp_count / len(df) * 100:.1f}%)"
    )

    return df


# =====================================================
# STEP 5 - CALCULATE SUMMARY METRICS
# =====================================================
def calculate_summary(df, month_str):
    """
    Calculate all KPIs for the monthly report.
    """
    log.info("Calculating monthly KPI summary")

    def safe_mean(col, default=0.0):
        if col in df.columns:
            return round(float(df[col].mean()), 3)
        return default

    def safe_pct(col, op, val, default=0.0):
        if col in df.columns:
            if op == '>=':
                return round(
                    float((df[col] >= val).mean() * 100), 1
                )
            elif op == '<=':
                return round(
                    float((df[col] <= val).mean() * 100), 1
                )
        return default

    def safe_sum(col, default=0):
        if col in df.columns:
            return int(df[col].sum())
        return default

    # Calculate risk flags inline
    flight_risk = int((
        (df.get('PerformanceRating',
                 pd.Series([3.0]*len(df))) >= 3.5) &
        (df.get('EngagementScore',
                 pd.Series([50.0]*len(df))) <= 45) &
        (df.get('YearsAtCompany',
                 pd.Series([2.0]*len(df))) >= 2.0)
    ).sum())

    pip_risk = int((
        (df['HighPerformerProbability'] <= 0.25) &
        (df.get('PerformanceRating',
                 pd.Series([3.0]*len(df))) <= 2.5)
    ).sum())

    promo_ready = int((
        df.get('PromotionReadinessScore',
               pd.Series([0.0]*len(df))) >= 0.65
    ).sum())

    summary = {
        'report_month'             : month_str,
        'generated_at'             : datetime.now().isoformat(),
        'total_employees'          : int(len(df)),
        'avg_performance_rating'   : safe_mean('PerformanceRating'),
        'pct_high_performers'      : safe_pct('PerformanceRating', '>=', 4.0),
        'pct_low_performers'       : safe_pct('PerformanceRating', '<=', 2.0),
        'avg_okr_completion'       : safe_mean('OKRCompletionPct'),
        'avg_engagement_score'     : safe_mean('EngagementScore'),
        'predicted_high_performers': int(df['PredictedHighPerformer'].sum()),
        'flight_risk_count'        : flight_risk,
        'pip_risk_count'           : pip_risk,
        'promotion_ready_count'    : promo_ready,
        'promotion_lag_count'      : safe_sum('PromotionLagFlag'),
    }

    log.info("Summary metrics:")
    for k, v in summary.items():
        if k not in ['generated_at', 'report_month']:
            log.info(f"   {k:<35}: {v}")

    return summary


# =====================================================
# STEP 6 - EXPORT OUTPUTS
# =====================================================
def export_outputs(df, summary, month_str, dry_run=False):
    """
    Save updated CSV for Power BI and summary JSON.
    """
    output_dir = PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        # Main CSV - Power BI reads this file
        csv_path = output_dir / 'employee_with_predictions.csv'
        df.to_csv(csv_path, index=False)
        log.info(f"Main CSV saved: {csv_path}")

        # Monthly archive
        archive_path = (
            output_dir / f'predictions_{month_str}.csv'
        )
        df.to_csv(archive_path, index=False)
        log.info(f"Archive saved: {archive_path}")

        # Summary JSON
        summary_path = (
            output_dir / f'summary_{month_str}.json'
        )
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        log.info(f"Summary JSON saved: {summary_path}")

    else:
        log.info("[DRY RUN] File saving skipped")

    return summary


# =====================================================
# STEP 7 - GENERATE MONTHLY REPORT
# =====================================================
def generate_report(df, summary, month_str, dry_run=False):
    """
    Generate a visual monthly report as PNG image.
    """
    log.info("Generating monthly visual report")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        C_BLUE   = '#2E4057'
        C_TEAL   = '#048A81'
        C_GREEN  = '#3BB273'
        C_RED    = '#E84855'
        C_ORANGE = '#EF8354'

        fig = plt.figure(figsize=(16, 20))
        fig.patch.set_facecolor('#FAFAFA')
        gs  = gridspec.GridSpec(
            5, 2, figure=fig,
            hspace=0.5, wspace=0.35
        )

        fig.suptitle(
            f"Monthly Performance Analytics Report\n"
            f"{month_str}  |  People Analytics Team",
            fontsize=18,
            fontweight='bold',
            y=0.98,
            color=C_BLUE
        )

        # --- KPI Banner ---
        ax0 = fig.add_subplot(gs[0, :])
        ax0.axis('off')
        kpi_text = (
            f"Employees: {summary['total_employees']:,}  |  "
            f"Avg Rating: {summary['avg_performance_rating']:.2f}/5  |  "
            f"High Performers: {summary['pct_high_performers']:.1f}%  |  "
            f"Avg OKR: {summary['avg_okr_completion']:.1f}%  |  "
            f"Avg Engagement: {summary['avg_engagement_score']:.1f}/100  |  "
            f"Promo Ready: {summary['promotion_ready_count']}  |  "
            f"Flight Risk: {summary['flight_risk_count']}"
        )
        ax0.text(
            0.5, 0.5, kpi_text,
            ha='center', va='center',
            fontsize=9,
            bbox=dict(
                boxstyle='round,pad=0.6',
                facecolor=C_BLUE,
                alpha=0.92
            ),
            color='white',
            transform=ax0.transAxes
        )
        ax0.set_title(
            "KEY METRICS THIS CYCLE",
            fontsize=11, fontweight='bold',
            color=C_BLUE, pad=8
        )

        # --- Chart 1: Rating Distribution ---
        if 'PerformanceRating' in df.columns:
            ax1 = fig.add_subplot(gs[1, 0])
            rc  = (df['PerformanceRating']
                   .value_counts()
                   .sort_index())
            bars = ax1.bar(
                rc.index.astype(str),
                rc.values,
                color=C_TEAL,
                edgecolor='white',
                linewidth=0.5
            )
            for bar, val in zip(bars, rc.values):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3,
                    f'{val/len(df)*100:.1f}%',
                    ha='center', fontsize=8
                )
            ax1.set_title(
                'Performance Rating Distribution',
                fontweight='bold', fontsize=10
            )
            ax1.set_xlabel('Rating')
            ax1.set_ylabel('Employees')
            ax1.spines[['top','right']].set_visible(False)

        # --- Chart 2: OKR Bands ---
        if 'OKRCompletionPct' in df.columns:
            ax2 = fig.add_subplot(gs[1, 1])
            okr_bands = pd.cut(
                df['OKRCompletionPct'],
                bins=[0, 50, 70, 85, 100],
                labels=['<50%','50-70%','70-85%','85%+']
            )
            oc = okr_bands.value_counts()
            ax2.pie(
                oc.values,
                labels=oc.index,
                colors=[C_RED, C_ORANGE, C_TEAL, C_GREEN],
                autopct='%1.1f%%',
                startangle=90,
                pctdistance=0.75,
                wedgeprops={'edgecolor':'white','linewidth':2}
            )
            ax2.set_title(
                'OKR Completion Bands',
                fontweight='bold', fontsize=10
            )

        # --- Chart 3: Dept Performance ---
        if ('Department' in df.columns and
                'PerformanceRating' in df.columns):
            ax3 = fig.add_subplot(gs[2, 0])
            da  = (df.groupby('Department')
                   ['PerformanceRating']
                   .mean()
                   .sort_values())
            ax3.barh(
                da.index, da.values,
                color=C_TEAL, edgecolor='white'
            )
            ax3.axvline(
                x=df['PerformanceRating'].mean(),
                color=C_RED, linestyle='--',
                linewidth=1.5,
                label=f"Avg: "
                      f"{df['PerformanceRating'].mean():.2f}"
            )
            ax3.set_title(
                'Avg Rating by Department',
                fontweight='bold', fontsize=10
            )
            ax3.legend(fontsize=8)
            ax3.spines[['top','right']].set_visible(False)

        # --- Chart 4: Burnout Risk ---
        if 'BurnoutRisk' in df.columns:
            ax4 = fig.add_subplot(gs[2, 1])
            bc  = (df['BurnoutRisk']
                   .value_counts()
                   .reindex(
                       ['Low','Medium','High'],
                       fill_value=0
                   ))
            ax4.bar(
                bc.index, bc.values,
                color=[C_GREEN, C_ORANGE, C_RED],
                edgecolor='white'
            )
            for i, val in enumerate(bc.values):
                ax4.text(
                    i, val + 2,
                    f'{val/len(df)*100:.1f}%',
                    ha='center', fontsize=9,
                    fontweight='bold'
                )
            ax4.set_title(
                'Burnout Risk Distribution',
                fontweight='bold', fontsize=10
            )
            ax4.spines[['top','right']].set_visible(False)

        # --- Chart 5: Risk Flags ---
        ax5 = fig.add_subplot(gs[3, :])
        risk_data = {
            'Flight Risk'     : summary['flight_risk_count'],
            'PIP Risk'        : summary['pip_risk_count'],
            'Promo Ready'     : summary['promotion_ready_count'],
            'Promo Lag'       : summary['promotion_lag_count'],
        }
        risk_colors = [C_ORANGE, C_RED, C_GREEN, C_TEAL]

        bars = ax5.bar(
            list(risk_data.keys()),
            list(risk_data.values()),
            color=risk_colors,
            edgecolor='white',
            width=0.5
        )
        for bar, val in zip(bars, risk_data.values()):
            pct = val / len(df) * 100
            ax5.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{val}\n({pct:.1f}%)',
                ha='center', fontsize=10,
                fontweight='bold'
            )
        ax5.set_title(
            'Monthly Action Items - Risk Flags',
            fontweight='bold', fontsize=11
        )
        ax5.spines[['top','right']].set_visible(False)

        # --- Top 10 Promotion Ready Table ---
        if 'PromotionReadinessScore' in df.columns:
            ax6 = fig.add_subplot(gs[4, :])
            ax6.axis('off')

            top_cols = [
                c for c in [
                    'EmployeeID', 'Department',
                    'JobLevel', 'PerformanceRating',
                    'PromotionReadinessScore',
                    'HighPerformerProbability'
                ] if c in df.columns
            ]

            top_10 = (
                df.nlargest(10, 'PromotionReadinessScore')
                  [top_cols]
                  .reset_index(drop=True)
            )

            tbl = ax6.table(
                cellText  = top_10.round(3).values,
                colLabels = top_cols,
                cellLoc   = 'center',
                loc       = 'center',
                bbox      = [0, 0, 1, 1]
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)

            for (row, col), cell in tbl.get_celld().items():
                if row == 0:
                    cell.set_facecolor(C_BLUE)
                    cell.set_text_props(
                        color='white',
                        fontweight='bold'
                    )
                elif row % 2 == 0:
                    cell.set_facecolor('#F0F4F8')
                cell.set_edgecolor('white')

            ax6.set_title(
                'Top 10 Promotion-Ready Employees',
                fontweight='bold', fontsize=11,
                pad=20
            )

        # --- Footer ---
        fig.text(
            0.5, 0.005,
            "Model-assisted insights. "
            "All decisions require human review. "
            "| People Analytics Automation System",
            ha='center', fontsize=8,
            color='#888888', style='italic'
        )

        # --- Save ---
        docs_dir    = PROJECT_ROOT / 'docs'
        docs_dir.mkdir(exist_ok=True)
        report_path = docs_dir / \
                      f'monthly_report_{month_str}.png'

        if not dry_run:
            plt.savefig(
                report_path,
                dpi=130,
                bbox_inches='tight',
                facecolor='#FAFAFA'
            )
            log.info(f"Report saved: {report_path}")
        else:
            log.info("[DRY RUN] Report not saved")

        plt.close()
        return str(report_path)

    except Exception as e:
        log.warning(f"Report generation error: {e}")
        return "report_generation_failed"


# =====================================================
# STEP 8 - MOCK EMAIL NOTIFICATION
# =====================================================
def send_notification(summary, report_path, month_str):
    """
    Portfolio demo: prints formatted email to console.
    Production: configure SMTP credentials in .env file.
    """
    subject = (
        f"[People Analytics] "
        f"Monthly Report Ready - {month_str}"
    )

    separator = "-" * 52

    body = (
        f"\n{separator}\n"
        f"PEOPLE ANALYTICS - MONTHLY AUTOMATED REPORT\n"
        f"{separator}\n\n"
        f"Hi HR Leadership Team,\n\n"
        f"Your monthly performance analytics report\n"
        f"for {month_str} is ready.\n\n"
        f"KEY HIGHLIGHTS:\n"
        f"{separator}\n"
        f"  Employees Analyzed        : "
        f"{summary['total_employees']:,}\n"
        f"  Avg Performance Rating    : "
        f"{summary['avg_performance_rating']:.2f} / 5.0\n"
        f"  High Performers (>=4.0)   : "
        f"{summary['pct_high_performers']:.1f}%\n"
        f"  Avg OKR Completion        : "
        f"{summary['avg_okr_completion']:.1f}%\n"
        f"  Avg Engagement Score      : "
        f"{summary['avg_engagement_score']:.1f} / 100\n"
        f"  Predicted High Performers : "
        f"{summary['predicted_high_performers']:,}\n"
        f"  Promotion Ready           : "
        f"{summary['promotion_ready_count']:,}\n\n"
        f"ACTIONS REQUIRED THIS CYCLE:\n"
        f"{separator}\n"
        f"  Flight Risk Employees     : "
        f"{summary['flight_risk_count']:,}\n"
        f"  -> Stay interviews within 2 weeks\n\n"
        f"  PIP Risk Employees        : "
        f"{summary['pip_risk_count']:,}\n"
        f"  -> PIP conversations within 30 days\n\n"
        f"  Promotion Lag Alerts      : "
        f"{summary['promotion_lag_count']:,}\n"
        f"  -> Career conversations within 30 days\n\n"
        f"REPORT ASSETS:\n"
        f"{separator}\n"
        f"  Report  : {report_path}\n"
        f"  Archive : "
        f"data/processed/predictions_{month_str}.csv\n"
        f"  Summary : "
        f"data/processed/summary_{month_str}.json\n\n"
        f"DISCLAIMER: All scores are model-assisted.\n"
        f"Final decisions require human review.\n\n"
        f"{separator}\n"
        f"People Analytics Automation System\n"
        f"Generated: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{separator}\n"
    )

    # ---- Uncomment below for production SMTP --------
    # from email.mime.multipart import MIMEMultipart
    # from email.mime.text import MIMEText
    # import smtplib
    # from dotenv import load_dotenv
    # load_dotenv()
    # msg = MIMEMultipart()
    # msg['Subject'] = subject
    # msg['From']    = os.getenv('EMAIL_FROM')
    # msg['To']      = os.getenv('EMAIL_TO')
    # msg.attach(MIMEText(body, 'plain'))
    # with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
    #     s.login(
    #         os.getenv('EMAIL_FROM'),
    #         os.getenv('EMAIL_PASSWORD')
    #     )
    #     s.send_message(msg)
    # log.info("Production email sent via SMTP")

    # ---- Demo mode: print to console ----------------
    log.info("")
    log.info("=" * 52)
    log.info("MOCK EMAIL NOTIFICATION (demo mode)")
    log.info(f"Subject: {subject}")
    log.info(body)
    log.info("=" * 52)


# =====================================================
# STEP 9 - POWER BI REFRESH NOTE
# =====================================================
def refresh_powerbi():
    """
    Prints Power BI refresh instructions.
    In production: use Power BI REST API.
    """
    log.info("")
    log.info("POWER BI REFRESH:")
    log.info("  CSV updated at: "
             "data/processed/employee_with_predictions.csv")
    log.info("  To refresh Power BI:")
    log.info("  1. Open Power BI Desktop")
    log.info("  2. Home tab -> Refresh")
    log.info("  OR configure scheduled refresh")
    log.info("  in Power BI Service (Settings -> Schedule)")


# =====================================================
# MAIN ORCHESTRATION
# =====================================================
def main():
    args      = parse_arguments()
    month_str = args.month
    dry_run   = args.dry_run
    start     = datetime.now()

    log.info("=" * 52)
    log.info("MONTHLY AUTOMATION STARTED")
    log.info(f"Month   : {month_str}")
    log.info(f"Dry Run : {dry_run}")
    log.info(f"Time    : {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 52)

    try:
        # Step 1: Load data
        df = load_monthly_data(month_str)

        # Step 2: Preprocess
        df, X_processed, feature_cols = preprocess_data(df)

        # Step 3: Predict
        df = generate_predictions(df, X_processed)

        # Step 4: Summary
        summary = calculate_summary(df, month_str)

        # Step 5: Export files
        export_outputs(df, summary, month_str, dry_run)

        # Step 6: Generate report
        report_path = generate_report(
            df, summary, month_str, dry_run
        )

        # Step 7: Send notification
        send_notification(summary, report_path, month_str)

        # Step 8: Power BI note
        refresh_powerbi()

        # Done
        elapsed = (datetime.now() - start).seconds
        log.info("")
        log.info("=" * 52)
        log.info("AUTOMATION COMPLETE")
        log.info(f"  Month    : {month_str}")
        log.info(f"  Duration : {elapsed} seconds")
        log.info(
            f"  HP Count : "
            f"{summary['predicted_high_performers']}"
        )
        log.info("  Manual hours saved: ~10 hours")
        log.info("=" * 52)

    except Exception as e:
        log.error("=" * 52)
        log.error("AUTOMATION FAILED")
        log.error(f"  Error: {e}")
        log.error("=" * 52)
        import traceback
        log.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()