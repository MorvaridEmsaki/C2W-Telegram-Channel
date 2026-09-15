"""
RQ3 Telegram posting activity analysis for thesis reporting.

This script examines how posting activity is associated with Telegram post
visibility and consultation inquiries. It covers daily posting frequency,
within-day post order, gaps between posts, recent posting density, and weekly
burstiness.

Usage:
    python rq3_posting_activity_analysis.py --input "Dataset 1_Posts.csv"

Outputs are written to:
    outputs/
    tables/
    figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
except ModuleNotFoundError as exc:
    missing_package = exc.name
    raise SystemExit(
        f"Missing required package: {missing_package}. "
        "Install the analysis dependencies with: "
        "pip install pandas numpy matplotlib scipy statsmodels openpyxl"
    ) from exc


TIME_PERIOD_ORDER = {
    "Morning": 0,
    "Afternoon": 1,
    "Evening": 2,
    "Night": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse Telegram posting frequency and order for RQ3."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the Telegram post-level CSV file.",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Base output directory. Defaults to the current directory.",
    )
    return parser.parse_args()


def make_output_dirs(base_dir: Path) -> dict[str, Path]:
    dirs = {
        "outputs": base_dir / "outputs",
        "tables": base_dir / "tables",
        "figures": base_dir / "figures",
    }
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    return dirs


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        date_col = "date"
    elif "date_tehran" in df.columns:
        date_col = "date_tehran"
    else:
        raise ValueError("The dataset must contain either 'date' or 'date_tehran'.")

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    if df["date"].isna().any():
        missing_dates = int(df["date"].isna().sum())
        raise ValueError(f"Could not parse {missing_dates} date values.")
    return df


def prepare_post_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = clean_column_names(df)
    df = parse_date_column(df)

    required_columns = [
        "id",
        "time_period",
        "views",
        "forwards",
        "consultation_inquiries",
        "fund",
        "country",
        "degree",
        "media_type",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_columns = [
        "id",
        "views",
        "forwards",
        "consultation_inquiries",
        "fund",
        "bachelor",
        "master",
        "phd",
        "post_doc",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["views"] = df["views"].replace(0, np.nan)
    df["cvr"] = df["consultation_inquiries"] / df["views"] * 1000
    df["forward_rate"] = df["forwards"] / df["views"] * 1000

    df["tp_ord"] = df["time_period"].map(TIME_PERIOD_ORDER)
    missing_periods = df.loc[df["tp_ord"].isna(), "time_period"].dropna().unique()
    if len(missing_periods) > 0:
        raise ValueError(f"Unexpected time period labels: {list(missing_periods)}")

    df = df.sort_values(["date", "tp_ord", "id"]).reset_index(drop=True)
    df["timestamp_approx"] = df["date"] + pd.to_timedelta(
        df["tp_ord"] * 6 + 3,
        unit="h",
    )

    df["posts_that_day"] = df.groupby("date")["cvr"].transform("size")
    df["pos_in_day"] = df.groupby("date").cumcount() + 1
    df["gap_h"] = df["timestamp_approx"].diff().dt.total_seconds() / 3600
    df["gap_days"] = df["date"].diff().dt.days

    daily_counts = df.groupby("date")["cvr"].size()
    full_daily_counts = daily_counts.reindex(
        pd.date_range(df["date"].min(), df["date"].max()),
        fill_value=0,
    )
    roll7 = full_daily_counts.rolling(7, min_periods=1).sum()
    roll14 = full_daily_counts.rolling(14, min_periods=1).sum()

    df["dens7"] = (
        df["date"].map(roll7)
        - df["posts_that_day"]
        + df["pos_in_day"]
        - 1
    )
    df["dens7_prev"] = df["date"].map(roll7.shift(1).fillna(0))
    df["dens14_prev"] = df["date"].map(roll14.shift(1).fillna(0))
    df["ym"] = df["date"].dt.to_period("M").astype(str)

    df["degree_defined"] = (df["degree"].fillna("Not Defined") != "Not Defined").astype(int)
    df["country_defined"] = (
        df["country"].fillna("Not Defined") != "Not Defined"
    ).astype(int)
    df["specificity"] = df["degree_defined"] * df["country_defined"]
    df["multi_level"] = df["degree"].fillna("").str.contains(";", regex=False).astype(int)
    df["media_label"] = df["media_type"].fillna("Unknown")

    df["gap_days_numeric"] = (df["gap_h"] / 24).fillna(0)
    df["gap_category"] = pd.cut(
        df["gap_h"],
        [-1, 0.1, 7, 13, 25, 49, 169, 1e6],
        labels=[
            "Same time slot",
            "<=6 h",
            "~12 h",
            "~1 day",
            "~2 days",
            "3-7 days",
            ">7 days",
        ],
    )
    df["cadence_band"] = pd.cut(
        df["posts_that_day"],
        [0, 1, 2, 4, 6, 20],
        labels=["1/day", "2/day", "3-4/day", "5-6/day", "7+/day"],
    )
    df["position_capped"] = df["pos_in_day"].clip(upper=6)

    return df


def build_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    day = (
        df.groupby("date")
        .agg(
            posts=("cvr", "size"),
            views=("views", "sum"),
            inquiries=("consultation_inquiries", "sum"),
            mean_views=("views", "mean"),
            mean_cvr=("cvr", "mean"),
            forwards=("forwards", "sum"),
            mean_forward_rate=("forward_rate", "mean"),
        )
        .reset_index()
    )
    day["cvr_pooled"] = day["inquiries"] / day["views"] * 1000
    day["inquiries_per_post"] = day["inquiries"] / day["posts"]
    day["ym"] = day["date"].dt.to_period("M").astype(str)
    day["t"] = (day["date"] - day["date"].min()).dt.days
    day["log_posts"] = np.log(day["posts"])
    day["log_inquiries"] = np.log(day["inquiries"].clip(lower=0.5))
    day["log_views"] = np.log(day["views"])
    day["log_mean_views"] = np.log(day["mean_views"])
    day["bucket"] = pd.cut(
        day["posts"],
        [0, 1, 2, 3, 4, 6, 8, 20],
        labels=["1", "2", "3", "4", "5-6", "7-8", "9+"],
    )

    for column in ["cvr_pooled", "mean_views", "inquiries_per_post", "inquiries"]:
        day[f"{column}_month_adjusted"] = (
            day[column]
            - day.groupby("ym")[column].transform("mean")
            + day[column].mean()
        )

    return day


def build_weekly_data(df: pd.DataFrame) -> pd.DataFrame:
    weekly = df.copy()
    weekly["week"] = weekly["date"].dt.to_period("W").dt.start_time
    week = (
        weekly.groupby("week")
        .agg(
            posts=("cvr", "size"),
            views=("views", "sum"),
            inquiries=("consultation_inquiries", "sum"),
            mean_views=("views", "mean"),
            active_days=("date", "nunique"),
        )
        .reset_index()
    )
    week["cvr"] = week["inquiries"] / week["views"] * 1000
    week["burstiness"] = week["posts"] / week["active_days"]
    week["inquiries_per_post"] = week["inquiries"] / week["posts"]
    week["ym"] = week["week"].dt.to_period("M").astype(str)
    week["log_posts"] = np.log(week["posts"])
    week["log_inquiries"] = np.log(week["inquiries"].clip(lower=0.5))
    week = week[week["posts"] >= 2].copy()
    week["weekly_band"] = pd.cut(
        week["posts"],
        [1, 3, 6, 10, 15, 60],
        labels=["2-3", "4-6", "7-10", "11-15", "16+"],
    )
    return week


def model_summary_table(model: object, terms: list[str], model_name: str) -> pd.DataFrame:
    rows = []
    conf_int = model.conf_int()
    for term in terms:
        if term not in model.params.index:
            continue
        rows.append(
            {
                "model": model_name,
                "term": term,
                "coefficient": model.params[term],
                "std_error": model.bse[term],
                "t_value": model.tvalues[term],
                "p_value": model.pvalues[term],
                "ci_lower_95": conf_int.loc[term, 0],
                "ci_upper_95": conf_int.loc[term, 1],
                "r_squared": model.rsquared,
                "n": int(model.nobs),
            }
        )
    return pd.DataFrame(rows)


def kruskal_result(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    label: str,
) -> dict[str, float | int | str]:
    groups = [
        group[value_col].dropna().values
        for _, group in data.groupby(group_col, observed=True)
        if len(group[value_col].dropna()) >= 2
    ]
    if len(groups) < 2:
        return {
            "test": label,
            "group_variable": group_col,
            "outcome": value_col,
            "h_statistic": np.nan,
            "p_value": np.nan,
            "number_of_groups": len(groups),
        }
    h_statistic, p_value = stats.kruskal(*groups)
    return {
        "test": label,
        "group_variable": group_col,
        "outcome": value_col,
        "h_statistic": round(float(h_statistic), 4),
        "p_value": round(float(p_value), 5),
        "number_of_groups": len(groups),
    }


def save_audit_tables(
    df: pd.DataFrame,
    day: pd.DataFrame,
    tables_dir: Path,
) -> None:
    audit = pd.DataFrame(
        {
            "metric": [
                "number_of_posts",
                "active_days",
                "first_date",
                "last_date",
                "duplicated_rows",
            ],
            "value": [
                len(df),
                len(day),
                df["date"].min().date().isoformat(),
                df["date"].max().date().isoformat(),
                int(df.duplicated().sum()),
            ],
        }
    )
    audit.to_excel(tables_dir / "Table_RQ3_Data_Audit.xlsx", index=False)

    day[
        ["posts", "views", "inquiries", "cvr_pooled", "mean_views", "inquiries_per_post"]
    ].describe().round(3).to_excel(
        tables_dir / "Table_RQ3_Daily_Descriptive_Statistics.xlsx"
    )

    df[
        ["posts_that_day", "pos_in_day", "gap_h", "dens7_prev", "dens14_prev", "cvr", "views"]
    ].describe().round(3).to_excel(
        tables_dir / "Table_RQ3_Post_Level_Descriptive_Statistics.xlsx"
    )


def save_daily_outputs(day: pd.DataFrame, tables_dir: Path) -> None:
    bucket_table = (
        day.groupby("bucket", observed=True)
        .agg(
            days=("posts", "size"),
            posts_per_day=("posts", "mean"),
            raw_cvr=("cvr_pooled", "mean"),
            adjusted_cvr=("cvr_pooled_month_adjusted", "mean"),
            raw_views_per_post=("mean_views", "mean"),
            adjusted_views_per_post=("mean_views_month_adjusted", "mean"),
            raw_inquiries_per_post=("inquiries_per_post", "mean"),
            adjusted_inquiries_per_post=("inquiries_per_post_month_adjusted", "mean"),
            total_inquiries=("inquiries", "mean"),
            adjusted_total_inquiries=("inquiries_month_adjusted", "mean"),
        )
        .round(3)
        .reset_index()
    )
    bucket_table.to_excel(
        tables_dir / "Table_RQ3_Daily_Frequency_Buckets.xlsx",
        index=False,
    )

    correlations = pd.DataFrame(
        [
            {
                "x": "posts_per_day",
                "y": "daily_pooled_cvr",
                "spearman_rho": stats.spearmanr(day["posts"], day["cvr_pooled"]).statistic,
                "p_value": stats.spearmanr(day["posts"], day["cvr_pooled"]).pvalue,
            },
            {
                "x": "posts_per_day",
                "y": "mean_views_per_post",
                "spearman_rho": stats.spearmanr(day["posts"], day["mean_views"]).statistic,
                "p_value": stats.spearmanr(day["posts"], day["mean_views"]).pvalue,
            },
            {
                "x": "posts_per_day",
                "y": "total_daily_inquiries",
                "spearman_rho": stats.spearmanr(day["posts"], day["inquiries"]).statistic,
                "p_value": stats.spearmanr(day["posts"], day["inquiries"]).pvalue,
            },
            {
                "x": "posts_per_day",
                "y": "inquiries_per_post",
                "spearman_rho": stats.spearmanr(day["posts"], day["inquiries_per_post"]).statistic,
                "p_value": stats.spearmanr(day["posts"], day["inquiries_per_post"]).pvalue,
            },
            {
                "x": "date",
                "y": "posts_per_day",
                "spearman_rho": stats.spearmanr(
                    day["date"].map(pd.Timestamp.toordinal),
                    day["posts"],
                ).statistic,
                "p_value": stats.spearmanr(
                    day["date"].map(pd.Timestamp.toordinal),
                    day["posts"],
                ).pvalue,
            },
        ]
    ).round(5)
    correlations.to_excel(
        tables_dir / "Table_RQ3_Daily_Spearman_Correlations.xlsx",
        index=False,
    )

    tests = pd.DataFrame(
        [
            kruskal_result(day, "bucket", "cvr_pooled", "Daily raw CVR by bucket"),
            kruskal_result(
                day,
                "bucket",
                "cvr_pooled_month_adjusted",
                "Daily within-month adjusted CVR by bucket",
            ),
        ]
    )
    tests.to_excel(tables_dir / "Table_RQ3_Daily_Nonparametric_Tests.xlsx", index=False)


def save_daily_models(day: pd.DataFrame, tables_dir: Path) -> None:
    model_specs = [
        (
            "D1 Efficiency, unadjusted",
            "cvr_pooled ~ posts",
            ["posts"],
        ),
        (
            "D2 Efficiency, month FE",
            "cvr_pooled ~ posts + C(ym)",
            ["posts"],
        ),
        (
            "D3 Reach per post, month FE",
            "log_mean_views ~ posts + C(ym)",
            ["posts"],
        ),
        (
            "D4 Total daily reach elasticity",
            "log_views ~ log_posts + C(ym)",
            ["log_posts"],
        ),
        (
            "D5 Total daily inquiries elasticity",
            "log_inquiries ~ log_posts + C(ym)",
            ["log_posts"],
        ),
        (
            "D6 Inquiries per post, month FE",
            "inquiries_per_post ~ posts + C(ym)",
            ["posts"],
        ),
    ]

    tables = []
    fitted_models = {}
    for name, formula, terms in model_specs:
        model = smf.ols(formula, data=day).fit(cov_type="HC3")
        fitted_models[name] = model
        tables.append(model_summary_table(model, terms, name))

    daily_models = pd.concat(tables, ignore_index=True).round(5)
    daily_models.to_excel(tables_dir / "Table_RQ3_Daily_Regression_Models.xlsx", index=False)

    inquiry_model = fitted_models["D5 Total daily inquiries elasticity"]
    reach_model = fitted_models["D4 Total daily reach elasticity"]

    elasticity_tests = []
    for name, model in [
        ("Inquiry elasticity vs 1", inquiry_model),
        ("Reach elasticity vs 1", reach_model),
    ]:
        coefficient = model.params["log_posts"]
        se = model.bse["log_posts"]
        z_value = (coefficient - 1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
        elasticity_tests.append(
            {
                "test": name,
                "coefficient": coefficient,
                "std_error": se,
                "z_value": z_value,
                "p_value": p_value,
            }
        )

    curvature_log = smf.ols(
        "log_inquiries ~ log_posts + I(log_posts**2) + C(ym)",
        data=day,
    ).fit(cov_type="HC3")
    curvature_count = smf.ols(
        "inquiries ~ posts + I(posts**2) + C(ym)",
        data=day,
    ).fit(cov_type="HC3")

    curvature_table = pd.concat(
        [
            model_summary_table(
                curvature_log,
                ["log_posts", "I(log_posts ** 2)"],
                "D7 Log inquiry curvature",
            ),
            model_summary_table(
                curvature_count,
                ["posts", "I(posts ** 2)"],
                "D8 Count inquiry curvature",
            ),
        ],
        ignore_index=True,
    ).round(5)
    curvature_table.to_excel(tables_dir / "Table_RQ3_Curvature_Models.xlsx", index=False)

    pd.DataFrame(elasticity_tests).round(5).to_excel(
        tables_dir / "Table_RQ3_Elasticity_Tests.xlsx",
        index=False,
    )

    b1 = curvature_log.params["log_posts"]
    b2 = curvature_log.params["I(log_posts ** 2)"]
    marginal_rows = []
    reference = b1 * np.log(1) + b2 * np.log(1) ** 2
    for n_posts in range(1, 13):
        current = np.exp((b1 * np.log(n_posts) + b2 * np.log(n_posts) ** 2) - reference)
        previous = (
            np.exp((b1 * np.log(n_posts - 1) + b2 * np.log(n_posts - 1) ** 2) - reference)
            if n_posts > 1
            else 0
        )
        marginal_rows.append(
            {
                "posts_per_day": n_posts,
                "index_total_inquiries": current,
                "marginal_gain": current - previous,
                "marginal_pct_of_first_post": 100 * (current - previous),
                "elasticity": b1 + 2 * b2 * np.log(n_posts),
            }
        )
    pd.DataFrame(marginal_rows).round(4).to_excel(
        tables_dir / "Table_RQ3_Marginal_Returns.xlsx",
        index=False,
    )


def save_post_level_outputs(df: pd.DataFrame, tables_dir: Path) -> None:
    position_table = (
        df.groupby("position_capped", observed=True)
        .agg(
            n=("cvr", "size"),
            raw_cvr=("cvr", "mean"),
            raw_views=("views", "mean"),
        )
        .round(3)
        .reset_index()
    )
    position_table.to_excel(tables_dir / "Table_RQ3_Post_Position.xlsx", index=False)

    gap_table = (
        df.groupby("gap_category", observed=True)
        .agg(
            n=("cvr", "size"),
            raw_cvr=("cvr", "mean"),
            raw_views=("views", "mean"),
        )
        .round(3)
        .reset_index()
    )
    gap_table.to_excel(tables_dir / "Table_RQ3_Gap_Categories.xlsx", index=False)

    df = df.copy()
    df["density_quintile"] = pd.qcut(
        df["dens7_prev"],
        5,
        labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        duplicates="drop",
    )
    density_table = (
        df.groupby("density_quintile", observed=True)
        .agg(
            n=("cvr", "size"),
            mean_prior_density=("dens7_prev", "mean"),
            raw_cvr=("cvr", "mean"),
            raw_views=("views", "mean"),
        )
        .round(3)
        .reset_index()
    )
    density_table.to_excel(tables_dir / "Table_RQ3_Prior_Density.xlsx", index=False)

    df["cvr_month_adjusted"] = (
        df["cvr"] - df.groupby("ym")["cvr"].transform("mean") + df["cvr"].mean()
    )
    df["views_month_adjusted"] = (
        df["views"] - df.groupby("ym")["views"].transform("mean") + df["views"].mean()
    )

    gap_adjusted_table = (
        df.groupby("gap_category", observed=True)
        .agg(
            n=("cvr", "size"),
            raw_cvr=("cvr", "mean"),
            adjusted_cvr=("cvr_month_adjusted", "mean"),
            raw_views=("views", "mean"),
            adjusted_views=("views_month_adjusted", "mean"),
        )
        .round(3)
        .reset_index()
    )
    gap_adjusted_table.to_excel(
        tables_dir / "Table_RQ3_Gap_Month_Adjusted.xlsx",
        index=False,
    )

    position_adjusted_table = (
        df.groupby("position_capped", observed=True)
        .agg(
            n=("cvr", "size"),
            raw_cvr=("cvr", "mean"),
            adjusted_cvr=("cvr_month_adjusted", "mean"),
            raw_views=("views", "mean"),
            adjusted_views=("views_month_adjusted", "mean"),
        )
        .round(3)
        .reset_index()
    )
    position_adjusted_table.to_excel(
        tables_dir / "Table_RQ3_Position_Month_Adjusted.xlsx",
        index=False,
    )

    post_model_specs = [
        (
            "P1 Cadence only",
            "cvr ~ C(cadence_band)",
            [
                "C(cadence_band)[T.2/day]",
                "C(cadence_band)[T.3-4/day]",
                "C(cadence_band)[T.5-6/day]",
                "C(cadence_band)[T.7+/day]",
            ],
        ),
        (
            "P2 Cadence + month FE",
            "cvr ~ C(cadence_band) + C(ym)",
            [
                "C(cadence_band)[T.2/day]",
                "C(cadence_band)[T.3-4/day]",
                "C(cadence_band)[T.5-6/day]",
                "C(cadence_band)[T.7+/day]",
            ],
        ),
        (
            "P3 Cadence + month FE + content controls",
            "cvr ~ C(cadence_band) + C(ym) + multi_level + specificity + fund + C(media_label)",
            [
                "C(cadence_band)[T.2/day]",
                "C(cadence_band)[T.3-4/day]",
                "C(cadence_band)[T.5-6/day]",
                "C(cadence_band)[T.7+/day]",
                "multi_level",
                "specificity",
                "fund",
            ],
        ),
        (
            "P4 Position in day",
            "cvr ~ pos_in_day + C(ym) + multi_level + specificity",
            ["pos_in_day"],
        ),
        (
            "P5 Gap since last post",
            "cvr ~ gap_days_numeric + C(ym) + multi_level + specificity",
            ["gap_days_numeric"],
        ),
        (
            "P6 7-day prior density",
            "cvr ~ dens7_prev + C(ym) + multi_level + specificity",
            ["dens7_prev"],
        ),
    ]

    model_tables = []
    for name, formula, terms in post_model_specs:
        model = smf.ols(formula, data=df).fit(cov_type="HC3")
        model_tables.append(model_summary_table(model, terms, name))

    pd.concat(model_tables, ignore_index=True).round(5).to_excel(
        tables_dir / "Table_RQ3_Post_Level_Regression_Models.xlsx",
        index=False,
    )

    tests = pd.DataFrame(
        [
            kruskal_result(df, "gap_category", "cvr", "Raw CVR by gap category"),
            kruskal_result(
                df,
                "gap_category",
                "cvr_month_adjusted",
                "Month-adjusted CVR by gap category",
            ),
            kruskal_result(
                df,
                "position_capped",
                "cvr_month_adjusted",
                "Month-adjusted CVR by within-day position",
            ),
        ]
    )
    tests.to_excel(tables_dir / "Table_RQ3_Post_Level_Nonparametric_Tests.xlsx", index=False)


def save_contrast_outputs(day: pd.DataFrame, tables_dir: Path) -> None:
    groups = {
        "1/day": day["bucket"] == "1",
        "2/day": day["bucket"] == "2",
        "3-4/day": day["bucket"].isin(["3", "4"]),
        "5-6/day": day["bucket"] == "5-6",
        "7+/day": day["bucket"].isin(["7-8", "9+"]),
    }
    base = day["cvr_pooled_month_adjusted"]
    rows = []
    labels = list(groups)
    for index, group_a in enumerate(labels):
        for group_b in labels[index + 1 :]:
            x = base[groups[group_a]].dropna()
            y = base[groups[group_b]].dropna()
            if len(x) < 2 or len(y) < 2:
                continue
            u_result = stats.mannwhitneyu(x, y, alternative="two-sided")
            pooled_sd = np.sqrt(
                ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1))
                / (len(x) + len(y) - 2)
            )
            rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": len(x),
                    "n_b": len(y),
                    "mean_a": x.mean(),
                    "mean_b": y.mean(),
                    "difference": x.mean() - y.mean(),
                    "cohens_d": (x.mean() - y.mean()) / pooled_sd,
                    "u_statistic": u_result.statistic,
                    "p_value": u_result.pvalue,
                }
            )

    contrasts = pd.DataFrame(rows)
    if not contrasts.empty:
        contrasts["p_holm"] = multipletests(
            contrasts["p_value"],
            method="holm",
        )[1]
        contrasts["significance"] = np.where(contrasts["p_holm"] < 0.05, "*", "ns")

    contrasts.round(4).to_excel(
        tables_dir / "Table_RQ3_Daily_Bucket_Contrasts.xlsx",
        index=False,
    )


def save_weekly_outputs(week: pd.DataFrame, tables_dir: Path) -> None:
    weekly_table = (
        week.groupby("weekly_band", observed=True)
        .agg(
            weeks=("posts", "size"),
            posts_per_week=("posts", "mean"),
            active_days=("active_days", "mean"),
            posts_per_active_day=("burstiness", "mean"),
            views_per_post=("mean_views", "mean"),
            cvr=("cvr", "mean"),
            inquiries_per_post=("inquiries_per_post", "mean"),
            total_inquiries=("inquiries", "mean"),
        )
        .round(3)
        .reset_index()
    )
    weekly_table.to_excel(tables_dir / "Table_RQ3_Weekly_Burstiness.xlsx", index=False)

    correlations = pd.DataFrame(
        [
            {
                "x": "posts_per_week",
                "y": "weekly_cvr",
                "spearman_rho": stats.spearmanr(week["posts"], week["cvr"]).statistic,
                "p_value": stats.spearmanr(week["posts"], week["cvr"]).pvalue,
            },
            {
                "x": "active_days",
                "y": "weekly_cvr",
                "spearman_rho": stats.spearmanr(week["active_days"], week["cvr"]).statistic,
                "p_value": stats.spearmanr(week["active_days"], week["cvr"]).pvalue,
            },
            {
                "x": "burstiness",
                "y": "weekly_cvr",
                "spearman_rho": stats.spearmanr(week["burstiness"], week["cvr"]).statistic,
                "p_value": stats.spearmanr(week["burstiness"], week["cvr"]).pvalue,
            },
        ]
    ).round(5)
    correlations.to_excel(
        tables_dir / "Table_RQ3_Weekly_Spearman_Correlations.xlsx",
        index=False,
    )

    weekly_tests = pd.DataFrame(
        [
            kruskal_result(
                week,
                "weekly_band",
                "cvr",
                "Weekly CVR by posting band",
            )
        ]
    )
    weekly_tests.to_excel(
        tables_dir / "Table_RQ3_Weekly_Nonparametric_Tests.xlsx",
        index=False,
    )

    model_specs = [
        (
            "W1 Weekly CVR by burstiness and volume",
            "cvr ~ burstiness + posts + C(ym)",
            ["burstiness", "posts"],
        ),
        (
            "W2 Weekly views per post by burstiness",
            "np.log(mean_views) ~ burstiness + posts + C(ym)",
            ["burstiness", "posts"],
        ),
        (
            "W3 Weekly inquiries elasticity and burstiness",
            "log_inquiries ~ log_posts + burstiness + C(ym)",
            ["log_posts", "burstiness"],
        ),
    ]
    model_tables = []
    for name, formula, terms in model_specs:
        model = smf.ols(formula, data=week).fit(cov_type="HC3")
        model_tables.append(model_summary_table(model, terms, name))

    pd.concat(model_tables, ignore_index=True).round(5).to_excel(
        tables_dir / "Table_RQ3_Weekly_Regression_Models.xlsx",
        index=False,
    )


def save_scenario_outputs(day: pd.DataFrame, tables_dir: Path) -> None:
    inquiry_curve = smf.ols(
        "log_inquiries ~ log_posts + I(log_posts**2) + C(ym)",
        data=day,
    ).fit(cov_type="HC3")
    reach_model = smf.ols(
        "log_mean_views ~ posts + C(ym)",
        data=day,
    ).fit(cov_type="HC3")

    b1 = inquiry_curve.params["log_posts"]
    b2 = inquiry_curve.params["I(log_posts ** 2)"]
    reach_decay = reach_model.params["posts"]

    scenario_rows = []
    for posts_per_day in [1, 2, 3, 4, 5, 6, 8, 10]:
        for active_days_per_week in [3, 5, 7]:
            index_daily = np.exp(
                b1 * np.log(posts_per_day)
                + b2 * np.log(posts_per_day) ** 2
            )
            previous_posts = max(posts_per_day - 1, 1)
            previous_index = np.exp(
                b1 * np.log(previous_posts)
                + b2 * np.log(previous_posts) ** 2
            )
            scenario_rows.append(
                {
                    "posts_per_day": posts_per_day,
                    "active_days_per_week": active_days_per_week,
                    "posts_per_week": posts_per_day * active_days_per_week,
                    "relative_inquiries_per_day": index_daily,
                    "relative_inquiries_per_week": index_daily * active_days_per_week,
                    "relative_views_per_post": np.exp(reach_decay * (posts_per_day - 1)),
                    "marginal_return_last_post": (
                        index_daily - previous_index if posts_per_day > 1 else 1.0
                    ),
                    "elasticity": b1 + 2 * b2 * np.log(posts_per_day),
                }
            )

    pd.DataFrame(scenario_rows).round(4).to_excel(
        tables_dir / "Table_RQ3_Cadence_Scenarios.xlsx",
        index=False,
    )


def plot_key_figures(day: pd.DataFrame, df: pd.DataFrame, figures_dir: Path) -> None:
    bucket_plot = (
        day.groupby("bucket", observed=True)["cvr_pooled_month_adjusted"]
        .mean()
        .dropna()
    )
    plt.figure(figsize=(8, 5))
    plt.bar(bucket_plot.index.astype(str), bucket_plot.values)
    plt.title("Month-Adjusted Inquiry Rate by Daily Posting Frequency")
    plt.xlabel("Posts per Active Day")
    plt.ylabel("Consultation Inquiries per 1,000 Views")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "Figure_RQ3_Adjusted_CVR_by_Daily_Frequency.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close()

    position_plot = (
        df.groupby("position_capped", observed=True)["cvr"]
        .mean()
        .dropna()
    )
    plt.figure(figsize=(7, 5))
    plt.bar(position_plot.index.astype(str), position_plot.values)
    plt.title("Inquiry Rate by Within-Day Post Position")
    plt.xlabel("Post Position in Day")
    plt.ylabel("Consultation Inquiries per 1,000 Views")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "Figure_RQ3_CVR_by_Post_Position.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close()


def save_cleaned_outputs(
    df: pd.DataFrame,
    day: pd.DataFrame,
    week: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    df.to_csv(outputs_dir / "rq3_post_level_cleaned.csv", index=False)
    day.to_csv(outputs_dir / "rq3_daily_level_cleaned.csv", index=False)
    week.to_csv(outputs_dir / "rq3_weekly_level_cleaned.csv", index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    base_dir = Path(args.outdir)
    dirs = make_output_dirs(base_dir)

    post_data = prepare_post_data(input_path)
    daily_data = build_daily_data(post_data)
    weekly_data = build_weekly_data(post_data)

    save_audit_tables(post_data, daily_data, dirs["tables"])
    save_daily_outputs(daily_data, dirs["tables"])
    save_daily_models(daily_data, dirs["tables"])
    save_contrast_outputs(daily_data, dirs["tables"])
    save_post_level_outputs(post_data, dirs["tables"])
    save_weekly_outputs(weekly_data, dirs["tables"])
    save_scenario_outputs(daily_data, dirs["tables"])
    plot_key_figures(daily_data, post_data, dirs["figures"])
    save_cleaned_outputs(post_data, daily_data, weekly_data, dirs["outputs"])

    print("RQ3 posting activity analysis completed successfully.")
    print(f"Post-level cleaned data: {dirs['outputs'] / 'rq3_post_level_cleaned.csv'}")
    print(f"Daily-level cleaned data: {dirs['outputs'] / 'rq3_daily_level_cleaned.csv'}")
    print(f"Weekly-level cleaned data: {dirs['outputs'] / 'rq3_weekly_level_cleaned.csv'}")
    print(f"Tables: {dirs['tables']}")
    print(f"Figures: {dirs['figures']}")


if __name__ == "__main__":
    main()
