"""
RQ1 Telegram post analysis for thesis reporting.

This script analyses how Telegram post characteristics are associated with
consultation inquiries. It uses both absolute inquiry counts and an
exposure-adjusted inquiry rate per 1,000 views, so the code matches the
methodological explanation in the thesis.

Usage:
    python rq1_telegram_analysis.py --input "Dataset 1_Posts.csv"

Outputs are written to:
    outputs/
    tables/
    figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy import stats
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ModuleNotFoundError as exc:
    missing_package = exc.name
    raise SystemExit(
        f"Missing required package: {missing_package}. "
        "Install the analysis dependencies with: "
        "pip install pandas numpy matplotlib scipy statsmodels openpyxl"
    ) from exc


DEPENDENT_COUNT = "consultation_inquiries"
DEPENDENT_RATE = "inquiry_rate_per_1000_views"
COUNT_EXPOSURE = "views"
FORWARD_RATE = "forward_rate_per_1000_views"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse Telegram content characteristics for RQ1."
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
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=10,
        help="Minimum category size before small groups are labelled as Other.",
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


def prepare_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = clean_column_names(df)

    required_columns = [
        "views",
        "forwards",
        "consultation_inquiries",
        "country",
        "degree",
        "fund",
        "qs_rank",
        "qs_available",
        "media_type",
        "bachelor",
        "master",
        "phd",
        "post_doc",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_columns = [
        "views",
        "forwards",
        "consultation_inquiries",
        "fund",
        "qs_rank",
        "qs_available",
        "bachelor",
        "master",
        "phd",
        "post_doc",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "weekday_number" in df.columns:
        df["weekday_number"] = pd.to_numeric(
            df["weekday_number"], errors="coerce"
        ).astype("Int64")

    df["views"] = df["views"].replace(0, np.nan)
    df[DEPENDENT_RATE] = (
        df["consultation_inquiries"] / df["views"] * 1000
    )
    df[FORWARD_RATE] = df["forwards"] / df["views"] * 1000

    return df


def group_small_categories(
    series: pd.Series,
    min_group_size: int,
    other_label: str = "Other",
) -> pd.Series:
    counts = series.value_counts(dropna=False)
    major_categories = counts[counts >= min_group_size].index
    return series.where(series.isin(major_categories), other_label)


def save_basic_audit(df: pd.DataFrame, tables_dir: Path) -> None:
    audit = pd.DataFrame(
        {
            "metric": [
                "number_of_rows",
                "number_of_columns",
                "duplicated_rows",
                "missing_text",
                "missing_qs_rank",
            ],
            "value": [
                df.shape[0],
                df.shape[1],
                int(df.duplicated().sum()),
                int(df["text"].isna().sum()) if "text" in df.columns else np.nan,
                int(df["qs_rank"].isna().sum()),
            ],
        }
    )
    audit.to_excel(tables_dir / "Table_RQ1_Data_Audit.xlsx", index=False)

    descriptive = df[
        [
            "consultation_inquiries",
            "views",
            "forwards",
            DEPENDENT_RATE,
            FORWARD_RATE,
            "qs_rank",
        ]
    ].describe().round(3)
    descriptive.to_excel(tables_dir / "Table_RQ1_Descriptive_Statistics.xlsx")


def group_descriptive_table(
    df: pd.DataFrame,
    group_col: str,
    output_path: Path,
) -> pd.DataFrame:
    table = (
        df.groupby(group_col, dropna=False)
        .agg(
            number_of_posts=(group_col, "count"),
            mean_consultation_inquiries=("consultation_inquiries", "mean"),
            median_consultation_inquiries=("consultation_inquiries", "median"),
            mean_inquiry_rate=(DEPENDENT_RATE, "mean"),
            median_inquiry_rate=(DEPENDENT_RATE, "median"),
            mean_views=("views", "mean"),
            median_views=("views", "median"),
            mean_forwards=("forwards", "mean"),
            median_forwards=("forwards", "median"),
            mean_forward_rate=(FORWARD_RATE, "mean"),
            median_forward_rate=(FORWARD_RATE, "median"),
        )
        .round(3)
        .sort_values("number_of_posts", ascending=False)
    )
    table.to_excel(output_path)
    return table


def degree_descriptive_table(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    degree_columns = ["bachelor", "master", "phd", "post_doc"]
    rows = []

    for degree in degree_columns:
        degree_posts = df[df[degree] == 1]
        rows.append(
            {
                "degree": degree.replace("_", " ").title(),
                "number_of_posts": len(degree_posts),
                "mean_consultation_inquiries": degree_posts[
                    "consultation_inquiries"
                ].mean(),
                "median_consultation_inquiries": degree_posts[
                    "consultation_inquiries"
                ].median(),
                "mean_inquiry_rate": degree_posts[DEPENDENT_RATE].mean(),
                "median_inquiry_rate": degree_posts[DEPENDENT_RATE].median(),
                "mean_views": degree_posts["views"].mean(),
                "mean_forwards": degree_posts["forwards"].mean(),
                "mean_forward_rate": degree_posts[FORWARD_RATE].mean(),
            }
        )

    table = pd.DataFrame(rows).round(3)
    table.to_excel(output_path, index=False)
    return table


def plot_bar(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    figsize: tuple[int, int] = (9, 5),
) -> None:
    plt.figure(figsize=figsize)
    bars = plt.bar(series.index.astype(str), series.values)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()


def create_descriptive_outputs(
    df: pd.DataFrame,
    dirs: dict[str, Path],
    min_group_size: int,
) -> pd.DataFrame:
    df = df.copy()
    df["country_grouped"] = group_small_categories(
        df["country"], min_group_size
    )
    df["media_type_grouped"] = group_small_categories(
        df["media_type"], min_group_size
    )

    group_descriptive_table(
        df,
        "country_grouped",
        dirs["tables"] / "Table_RQ1_Country_Statistics.xlsx",
    )
    group_descriptive_table(
        df,
        "fund",
        dirs["tables"] / "Table_RQ1_Funding_Statistics.xlsx",
    )
    group_descriptive_table(
        df,
        "media_type_grouped",
        dirs["tables"] / "Table_RQ1_Media_Type_Statistics.xlsx",
    )
    degree_table = degree_descriptive_table(
        df,
        dirs["tables"] / "Table_RQ1_Degree_Level_Statistics.xlsx",
    )

    chart_specs = [
        ("country_grouped", DEPENDENT_RATE, "Average Inquiry Rate by Country"),
        ("country_grouped", "views", "Average Views by Country"),
        ("country_grouped", FORWARD_RATE, "Average Forward Rate by Country"),
        ("fund", DEPENDENT_RATE, "Average Inquiry Rate by Funding Status"),
        ("fund", "views", "Average Views by Funding Status"),
        ("fund", FORWARD_RATE, "Average Forward Rate by Funding Status"),
        ("media_type_grouped", DEPENDENT_RATE, "Average Inquiry Rate by Media Type"),
        ("media_type_grouped", "views", "Average Views by Media Type"),
        ("media_type_grouped", FORWARD_RATE, "Average Forward Rate by Media Type"),
    ]

    for group_col, value_col, title in chart_specs:
        chart = df.groupby(group_col)[value_col].mean().sort_values(ascending=False)
        plot_bar(
            chart,
            title,
            group_col.replace("_", " ").title(),
            value_col.replace("_", " ").title(),
            dirs["figures"] / f"{title.replace(' ', '_')}.svg",
        )

    for value_col, title in [
        ("mean_inquiry_rate", "Average Inquiry Rate by Degree Level"),
        ("mean_views", "Average Views by Degree Level"),
        ("mean_forward_rate", "Average Forward Rate by Degree Level"),
    ]:
        chart = degree_table.set_index("degree")[value_col].sort_values(
            ascending=False
        )
        plot_bar(
            chart,
            title,
            "Degree Level",
            value_col.replace("_", " ").title(),
            dirs["figures"] / f"{title.replace(' ', '_')}.svg",
            figsize=(7, 5),
        )

    return df


def save_distribution_plots(df: pd.DataFrame, figures_dir: Path) -> None:
    variables = [
        ("consultation_inquiries", "Distribution of Consultation Inquiries"),
        (DEPENDENT_RATE, "Distribution of Inquiry Rate per 1,000 Views"),
        ("views", "Distribution of Views"),
        ("forwards", "Distribution of Forwards"),
        (FORWARD_RATE, "Distribution of Forward Rate per 1,000 Views"),
        ("qs_rank", "Distribution of QS University Rank"),
    ]

    for column, title in variables:
        data = df[column].dropna()
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=20, edgecolor="black")
        plt.title(title)
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{title.replace(' ', '_')}.svg", format="svg")
        plt.close()

    plt.figure(figsize=(6, 6))
    stats.probplot(df[DEPENDENT_RATE].dropna(), dist="norm", plot=plt)
    plt.title("Q-Q Plot of Inquiry Rate per 1,000 Views")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "QQ_Plot_Inquiry_Rate_per_1000_Views.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close()


def spearman_pair(
    df: pd.DataFrame,
    x: str,
    y: str,
) -> dict[str, float | int | str]:
    data = df[[x, y]].dropna()
    rho, p_value = stats.spearmanr(data[x], data[y])
    return {
        "x": x,
        "y": y,
        "spearman_rho": round(float(rho), 3),
        "p_value": round(float(p_value), 4),
        "n": int(len(data)),
    }


def save_correlation_outputs(df: pd.DataFrame, tables_dir: Path) -> None:
    pairs = [
        ("qs_rank", "consultation_inquiries"),
        ("qs_rank", DEPENDENT_RATE),
        ("qs_rank", "views"),
        ("qs_rank", "forwards"),
        ("views", "consultation_inquiries"),
        ("views", DEPENDENT_RATE),
        ("forwards", "consultation_inquiries"),
        (FORWARD_RATE, DEPENDENT_RATE),
        ("views", "forwards"),
    ]

    pair_table = pd.DataFrame([spearman_pair(df, x, y) for x, y in pairs])
    pair_table.to_excel(
        tables_dir / "Table_RQ1_Spearman_Pairwise_Correlations.xlsx",
        index=False,
    )

    matrix_vars = [
        "views",
        "forwards",
        "consultation_inquiries",
        DEPENDENT_RATE,
        FORWARD_RATE,
        "qs_rank",
    ]
    matrix = df[matrix_vars].corr(method="spearman").round(3)
    matrix.to_excel(tables_dir / "Table_RQ1_Spearman_Correlation_Matrix.xlsx")


def kruskal_test(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    min_group_size: int = 2,
) -> dict[str, float | int | str]:
    groups = [
        group[value_col].dropna().values
        for _, group in df.groupby(group_col, dropna=False)
        if len(group[value_col].dropna()) >= min_group_size
    ]

    if len(groups) < 2:
        return {
            "group_variable": group_col,
            "outcome": value_col,
            "test": "Kruskal-Wallis",
            "h_statistic": np.nan,
            "p_value": np.nan,
            "number_of_groups": len(groups),
        }

    h_statistic, p_value = stats.kruskal(*groups)
    return {
        "group_variable": group_col,
        "outcome": value_col,
        "test": "Kruskal-Wallis",
        "h_statistic": round(float(h_statistic), 3),
        "p_value": round(float(p_value), 4),
        "number_of_groups": len(groups),
    }


def mann_whitney_test(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> dict[str, float | int | str]:
    values = sorted(df[group_col].dropna().unique())
    if len(values) != 2:
        return {
            "group_variable": group_col,
            "outcome": value_col,
            "test": "Mann-Whitney U",
            "u_statistic": np.nan,
            "p_value": np.nan,
            "number_of_groups": len(values),
        }

    group_1 = df[df[group_col] == values[0]][value_col].dropna()
    group_2 = df[df[group_col] == values[1]][value_col].dropna()
    u_statistic, p_value = stats.mannwhitneyu(
        group_1,
        group_2,
        alternative="two-sided",
    )
    return {
        "group_variable": group_col,
        "outcome": value_col,
        "test": "Mann-Whitney U",
        "u_statistic": round(float(u_statistic), 3),
        "p_value": round(float(p_value), 4),
        "number_of_groups": 2,
        "group_1": values[0],
        "group_2": values[1],
    }


def save_group_test_outputs(df: pd.DataFrame, tables_dir: Path) -> None:
    outcomes = [
        "consultation_inquiries",
        DEPENDENT_RATE,
        "views",
        "forwards",
        FORWARD_RATE,
    ]
    multi_group_variables = [
        "country_grouped",
        "media_type_grouped",
        "degree",
    ]
    binary_group_variables = [
        "fund",
        "qs_available",
        "has_media",
        "bachelor",
        "master",
        "phd",
        "post_doc",
    ]
    binary_group_variables = [
        column for column in binary_group_variables if column in df.columns
    ]

    rows = []
    for outcome in outcomes:
        for group_col in multi_group_variables:
            rows.append(kruskal_test(df, group_col, outcome))
        for group_col in binary_group_variables:
            rows.append(mann_whitney_test(df, group_col, outcome))

    pd.DataFrame(rows).to_excel(
        tables_dir / "Table_RQ1_Nonparametric_Group_Tests.xlsx",
        index=False,
    )


def add_region_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    region_map = {
        "Canada": "North America",
        "United States": "North America",
        "United Kingdom": "Europe",
        "Sweden": "Europe",
        "Italy": "Europe",
        "Netherlands": "Europe",
        "Spain": "Europe",
        "Denmark": "Europe",
        "Ireland": "Europe",
        "Germany": "Europe",
        "France": "Europe",
        "Austria": "Europe",
        "Belgium": "Europe",
        "Norway": "Europe",
        "Switzerland": "Europe",
        "Romania": "Europe",
        "Poland": "Europe",
        "Hungary": "Europe",
        "Finland": "Europe",
        "Australia": "Oceania",
        "New Zealand": "Oceania",
        "Japan": "Asia",
        "Taiwan": "Asia",
        "Singapore": "Asia",
        "China": "Asia",
        "Turkey": "Asia",
        "Malaysia": "Asia",
        "South Korea": "Asia",
        "Thailand": "Asia",
        "Qatar": "Middle East",
        "UAE": "Middle East",
        "Saudi Arabia": "Middle East",
        "Not Defined": "Not Defined",
    }
    df["country_region"] = df["country"].map(region_map).fillna("Other")
    return df


def prepare_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    df = add_region_groups(df)
    model_data = df[
        [
            "consultation_inquiries",
            DEPENDENT_RATE,
            "views",
            "fund",
            "bachelor",
            "master",
            "phd",
            "post_doc",
            "qs_available",
            "country_region",
            "media_type_grouped",
        ]
    ].dropna(subset=["consultation_inquiries", "views"]).copy()

    model_data["log_views"] = np.log(model_data["views"])
    return model_data


def fit_ols_rate_model(model_data: pd.DataFrame) -> object:
    formula = (
        f"{DEPENDENT_RATE} ~ fund + bachelor + master + phd + post_doc "
        "+ qs_available + C(country_region) + C(media_type_grouped)"
    )
    return smf.ols(formula=formula, data=model_data).fit(cov_type="HC3")


def fit_negative_binomial_count_model(model_data: pd.DataFrame) -> object:
    formula = (
        "consultation_inquiries ~ fund + bachelor + master + phd + post_doc "
        "+ qs_available + C(country_region) + C(media_type_grouped)"
    )
    return smf.glm(
        formula=formula,
        data=model_data,
        family=sm.families.NegativeBinomial(),
        offset=model_data["log_views"],
    ).fit(cov_type="HC3")


def model_summary_table(model: object) -> pd.DataFrame:
    conf_int = model.conf_int()
    table = pd.DataFrame(
        {
            "coefficient": model.params,
            "std_error": model.bse,
            "test_statistic": model.tvalues
            if hasattr(model, "tvalues")
            else model.zvalues,
            "p_value": model.pvalues,
            "ci_lower_95": conf_int.iloc[:, 0]
            if hasattr(conf_int, "iloc")
            else conf_int[:, 0],
            "ci_upper_95": conf_int.iloc[:, 1]
            if hasattr(conf_int, "iloc")
            else conf_int[:, 1],
        }
    )
    return table.round(4)


def save_regression_outputs(df: pd.DataFrame, dirs: dict[str, Path]) -> None:
    model_data = prepare_regression_data(df)

    ols_rate_model = fit_ols_rate_model(model_data)
    nb_count_model = fit_negative_binomial_count_model(model_data)

    model_summary_table(ols_rate_model).to_excel(
        dirs["tables"] / "Table_RQ1_OLS_Rate_Model_HC3.xlsx"
    )
    model_summary_table(nb_count_model).to_excel(
        dirs["tables"] / "Table_RQ1_Negative_Binomial_Count_Model_HC3.xlsx"
    )

    model_fit = pd.DataFrame(
        [
            {
                "model": "OLS rate model with HC3 robust standard errors",
                "n": int(ols_rate_model.nobs),
                "r_squared": round(float(ols_rate_model.rsquared), 4),
                "adj_r_squared": round(float(ols_rate_model.rsquared_adj), 4),
                "aic": round(float(ols_rate_model.aic), 3),
                "bic": round(float(ols_rate_model.bic), 3),
            },
            {
                "model": "Negative binomial count model with log(views) offset",
                "n": int(nb_count_model.nobs),
                "pseudo_r_squared_cs": round(
                    float(nb_count_model.pseudo_rsquared(kind="cs")), 4
                ),
                "aic": round(float(nb_count_model.aic), 3),
                "bic": round(float(nb_count_model.bic_llf), 3),
            },
        ]
    )
    model_fit.to_excel(dirs["tables"] / "Table_RQ1_Model_Fit_Summary.xlsx", index=False)

    design_matrix = pd.get_dummies(
        model_data[
            [
                "fund",
                "bachelor",
                "master",
                "phd",
                "post_doc",
                "qs_available",
                "country_region",
                "media_type_grouped",
            ]
        ],
        drop_first=True,
        dtype=int,
    )
    vif_table = pd.DataFrame(
        {
            "variable": design_matrix.columns,
            "vif": [
                variance_inflation_factor(design_matrix.values, i)
                for i in range(design_matrix.shape[1])
            ],
        }
    ).sort_values("vif", ascending=False)
    vif_table.round(3).to_excel(dirs["tables"] / "Table_RQ1_VIF.xlsx", index=False)

    residuals = ols_rate_model.resid
    fitted_values = ols_rate_model.fittedvalues
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_values, residuals, alpha=0.5)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted Values: OLS Inquiry Rate Model")
    plt.tight_layout()
    plt.savefig(
        dirs["figures"] / "Residuals_vs_Fitted_OLS_Inquiry_Rate_Model.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close()


def save_cleaned_dataset(df: pd.DataFrame, outputs_dir: Path) -> None:
    df.to_csv(outputs_dir / "telegram_cleaned_rq1_analysis.csv", index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    base_dir = Path(args.outdir)
    dirs = make_output_dirs(base_dir)

    df = prepare_data(input_path)
    save_basic_audit(df, dirs["tables"])

    df = create_descriptive_outputs(
        df,
        dirs,
        min_group_size=args.min_group_size,
    )
    save_distribution_plots(df, dirs["figures"])
    save_correlation_outputs(df, dirs["tables"])
    save_group_test_outputs(df, dirs["tables"])
    save_regression_outputs(df, dirs)
    save_cleaned_dataset(df, dirs["outputs"])

    print("RQ1 analysis completed successfully.")
    print(f"Cleaned data: {dirs['outputs'] / 'telegram_cleaned_rq1_analysis.csv'}")
    print(f"Tables: {dirs['tables']}")
    print(f"Figures: {dirs['figures']}")


if __name__ == "__main__":
    main()
