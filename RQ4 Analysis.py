"""
================================================================================
RQ4 - ANALYSIS CODE
IEP Canada Telegram channel, Master's thesis

Sub-research question 4:
    How is posting timing associated with content visibility and consultation
    inquiry generation?

Definitions used throughout
    Visibility        = views on the post
    Consultation inquiry = user direct message/inquiry attributed to the post.
    Inquiry rate         = (consultation inquiries / views) * 1,000

Timing dimensions
    period  : time-of-day block (Morning, Afternoon, Evening, Night)
    weekday : day of week
    hour    : retained for the robustness check on a finer grid

Procedures
    1. Descriptives by period and by weekday
    2. Visibility: OLS on log(views), HC3 robust SE, fitted twice -
       unadjusted, then adjusted for channel era and content mix;
       significance judged on a JOINT F test of the period block
    3. Inquiry generation: NB2 on the consultation inquiry count with log(views) as an offset,
       dispersion estimated by ML, HC1 robust SE; joint Wald chi-square on the
       period block
    4. Weekday models, same structure
    5. Robustness: hour-of-day grid, and the channel-era figure

Headline result recorded in the thesis
    The raw Evening and Night visibility difference does not
    survive adjustment for channel era and content mix (joint F = 1.81,
    p = .144). The inquiry-rate model is a precise null (joint chi-square = 0.14,
    p = .986). Weekday is null throughout. Reported as "no statistically
    reliable association was detected", not as evidence of no relationship at
    any size.

Reproducibility note
    Set DATA_PATH and, if your column headers differ, adjust COLS below.
    Requires: pandas, numpy, scipy, statsmodels.
================================================================================
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ------------------------------------------------------------------ CONFIG ---
DATA_PATH = "Dataset 1_Posts.csv"      # same post-level export used for RQ1 and RQ3
OUT_DIR = "."

COLS = {
    "views":   "views",
    "inquiries": "Consultation Inquiries",
    "date":      "date_tehran",
    "period":    "Time Period",
    "weekday":   "weekday_tehran",
    "region":    "Country",
    "degree":    "Degree",
    "funding":   "Fund",
    "media":     "media_type",
}

PERIOD_ORDER = ["Morning", "Afternoon", "Evening", "Night"]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

# Content-mix controls. These are the RQ1 characteristics that were shown to
# be associated with the outcome, so leaving them out would let a timing
# coefficient absorb the fact that different kinds of post go out at different
# hours.
CONTENT_MIX = "C(region) + degree_levels + C(funding) + C(media)"


def to_period(hour):
    """Time-of-day blocks. Boundaries fixed before looking at the outcomes."""
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 18:
        return "Afternoon"
    if 18 <= hour < 24:
        return "Evening"
    return "Night"


def count_degree_levels(text):
    """Same ordinal degree-specificity measure as in RQ1."""
    t = str(text).lower()
    levels = 0
    for keys in (("bachelor", "undergrad", "bsc", "ba "),
                 ("master", "msc", "ma ", "mba"),
                 ("phd", "doctor", "dphil")):
        if any(k in t for k in keys):
            levels += 1
    return levels


# ---------------------------------------------------------------- LOAD -------
def load():
    raw = pd.read_excel(DATA_PATH) if DATA_PATH.lower().endswith(("xlsx", "xls")) \
        else pd.read_csv(DATA_PATH)
    df = raw.rename(columns={v: k for k, v in COLS.items() if v in raw.columns}).copy()

    required = ["views", "inquiries", "date", "region", "degree", "funding", "media"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after renaming: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["views"] = pd.to_numeric(df["views"], errors="coerce")
    df["inquiries"] = pd.to_numeric(df["inquiries"], errors="coerce").fillna(0)
    df = df[(df["views"] > 0) & df["date"].notna()].copy()

    if "period" in df.columns:
        df["period"] = df["period"].astype(str).str.strip()
        df.loc[~df["period"].isin(PERIOD_ORDER), "period"] = np.nan
    else:
        df["hour"] = df["date"].dt.hour
        df["period"] = df["hour"].apply(to_period)
    if "hour" not in df.columns:
        approx_hour = {"Morning": 9, "Afternoon": 15, "Evening": 21, "Night": 3}
        df["hour"] = df["period"].map(approx_hour)
    df["period"] = pd.Categorical(df["period"], PERIOD_ORDER, ordered=False)

    if "weekday" in df.columns:
        df["weekday"] = df["weekday"].astype(str).str.strip()
    else:
        df["weekday"] = df["date"].dt.day_name()
    df["weekday"] = pd.Categorical(df["weekday"], WEEKDAY_ORDER, ordered=False)

    # Channel era. Audience size, posting volume and consultation inquiry volume all drift
    # across the window, so month is carried as a fixed effect in every
    # adjusted model.
    df["month"] = df["date"].dt.to_period("M").astype(str)

    for c in ("region", "degree", "funding", "media"):
        df[c] = df[c].astype(str).str.strip().replace({"": "Not specified",
                                                       "nan": "Not specified"})
    df["degree_levels"] = df["degree"].apply(count_degree_levels)

    df["inquiry_rate"] = 1000.0 * df["inquiries"] / df["views"]
    df["log_views"] = np.log(df["views"])
    return df


# ------------------------------------------------------- 1. DESCRIPTIVES -----
def descriptives(df):
    """Median and mean visibility and inquiry rate by period and by weekday."""
    out = {}
    for key, order in [("period", PERIOD_ORDER), ("weekday", WEEKDAY_ORDER)]:
        t = (df.groupby(key, observed=True)
               .agg(N=("views", "size"),
                    Median_views=("views", "median"),
                    Mean_views=("views", "mean"),
                    Total_inquiries=("inquiries", "sum"),
                    Inquiry_rate=("inquiry_rate", "mean"))
               .reindex(order).reset_index())
        # Raw percentage difference in median views against the busiest block,
        # quoted in the thesis only to be shown as an era artefact afterwards.
        base = t["Median_views"].iloc[0]
        t["Pct_vs_first_block"] = (100 * (t["Median_views"] / base - 1)).round(1)
        t.to_csv(f"{OUT_DIR}/RQ4_T1_descriptives_{key}.csv", index=False)
        print(t, "\n")
        out[key] = t
    return out


# ------------------------------------------------------- 2. VISIBILITY -------
def visibility_models(df, timing="period"):
    """
    OLS on log(views). A log outcome means each coefficient converts to a
    percentage difference:  (exp(b) - 1) * 100.

    Two specifications:
      unadjusted : log(views) ~ timing
      adjusted   : log(views) ~ timing + month fixed effects + content mix

    The reported test is the JOINT F test of the whole timing block, not the
    individual coefficients, so that one block crossing p = .05 by itself is
    not read as a timing effect.
    """
    results = {}
    specs = {
        "unadjusted": f"log_views ~ C({timing})",
        "adjusted": f"log_views ~ C({timing}) + C(month) + {CONTENT_MIX}",
    }
    rows = []
    for label, formula in specs.items():
        m = smf.ols(formula, data=df).fit(cov_type="HC3")
        results[label] = m

        terms = [t for t in m.params.index if t.startswith(f"C({timing})")]
        R = np.zeros((len(terms), len(m.params)))
        for i, t in enumerate(terms):
            R[i, list(m.params.index).index(t)] = 1
        F = m.f_test(R)

        for t in terms:
            rows.append({
                "Specification": label,
                "Term": t,
                "b": round(m.params[t], 4),
                "SE_HC3": round(m.bse[t], 4),
                "pct_difference": round((np.exp(m.params[t]) - 1) * 100, 1),
                "p": m.pvalues[t],
                "joint_F": round(float(np.squeeze(F.fvalue)), 3),
                "joint_p": float(np.squeeze(F.pvalue)),
                "N": int(m.nobs),
            })
        print(f"[{timing} / {label}] joint F = {float(np.squeeze(F.fvalue)):.3f}, "
              f"p = {float(np.squeeze(F.pvalue)):.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/RQ4_T2_visibility_{timing}.csv", index=False)
    return out, results


# --------------------------------------------- 3. INQUIRY GENERATION --------
def inquiry_models(df, timing="period"):
    """
    NB2 model of the consultation inquiry count with log(views) as an offset, so the
    coefficients describe inquiries PER VIEW - the inquiry rate - rather than raw
    consultation inquiry volume. A busy post generating more inquiries simply because more
    people saw it is therefore not counted as a timing effect.

    The dispersion parameter alpha is estimated by maximum likelihood and then
    held at that value in a GLM refit that supplies HC1 robust standard errors.
    HC1 guards against residual misspecification of the variance function; it
    does not substitute for estimating alpha correctly, which is why alpha is
    estimated rather than fixed.

    Reported as incidence rate ratios, with a joint Wald chi-square on the
    timing block.
    """
    formula = f"inquiries ~ C({timing}) + C(month) + {CONTENT_MIX}"
    offset = np.log(df["views"])

    # Is a Poisson model adequate? Pearson chi2/df plus a boundary-corrected
    # likelihood-ratio test (the null sits on the edge of the parameter space,
    # so the p-value is halved).
    pois = smf.glm(formula, data=df, family=sm.families.Poisson(),
                   offset=offset).fit()
    ml = smf.negativebinomial(formula, data=df,
                              offset=offset).fit(method="bfgs", maxiter=500, disp=0)
    LR = 2 * (ml.llf - pois.llf)
    print(f"Pearson chi2/df (Poisson) = {pois.pearson_chi2 / pois.df_resid:.3f}")
    print(f"LR Poisson vs NB2 = {LR:.2f}, "
          f"boundary-corrected p = {0.5 * st.chi2.sf(LR, 1):.4g}")

    alpha_hat = float(ml.params["alpha"])
    glm = smf.glm(formula, data=df,
                  family=sm.families.NegativeBinomial(alpha=alpha_hat),
                  offset=offset).fit(cov_type="HC1")
    print(glm.summary())

    terms = [t for t in glm.params.index if t.startswith(f"C({timing})")]
    R = np.zeros((len(terms), len(glm.params)))
    for i, t in enumerate(terms):
        R[i, list(glm.params.index).index(t)] = 1
    wald = glm.wald_test(R, scalar=True)
    print(f"[{timing}] joint Wald chi2 = {float(wald.statistic):.3f}, "
          f"p = {float(wald.pvalue):.4f}")

    out = pd.DataFrame({
        "Term": terms,
        "b": [glm.params[t] for t in terms],
        "SE_HC1": [glm.bse[t] for t in terms],
        "IRR": [np.exp(glm.params[t]) for t in terms],
        "IRR_lo": [np.exp(glm.params[t] - 1.96 * glm.bse[t]) for t in terms],
        "IRR_hi": [np.exp(glm.params[t] + 1.96 * glm.bse[t]) for t in terms],
        "p": [glm.pvalues[t] for t in terms],
    })
    out["alpha_ml"] = round(alpha_hat, 6)
    out["joint_wald_chi2"] = round(float(wald.statistic), 3)
    out["joint_p"] = float(wald.pvalue)
    out["N"] = int(glm.nobs)
    out.to_csv(f"{OUT_DIR}/RQ4_T3_inquiryrate_{timing}.csv", index=False)
    return out, glm


# --------------------------------- 4. WHY THE RAW TIMING GAP DISAPPEARS -------
def era_and_mix_diagnostics(df):
    """
    Two descriptive checks that explain the gap between the raw and the
    adjusted visibility results.

    (a) Posting time shifted across the observation window, so a time block is
        partly a proxy for a period in the channel's life.
    (b) Different kinds of content go out at different times, so a time block
        is also partly a proxy for content mix.
    """
    era = pd.crosstab(df["month"], df["period"], normalize="index").round(3)
    era.to_csv(f"{OUT_DIR}/RQ4_T4_period_share_by_month.csv")

    mix = (df.groupby("period", observed=True)
             .agg(mean_degree_levels=("degree_levels", "mean"),
                  pct_funding=("funding",
                               lambda s: 100 * (~s.str.contains("no|none|not",
                                                                case=False)).mean()),
                  median_views=("views", "median"))
             .reindex(PERIOD_ORDER).round(3))
    mix.to_csv(f"{OUT_DIR}/RQ4_T5_content_mix_by_period.csv")

    monthly_views = (df.groupby("month")["views"].median()
                       .rename("Median_views").reset_index())
    monthly_views.to_csv(f"{OUT_DIR}/RQ4_T6_channel_era.csv", index=False)

    print(era, "\n"); print(mix, "\n")
    return era, mix, monthly_views


# ------------------------------------------ 5. ROBUSTNESS: HOUR-OF-DAY GRID --
def hour_robustness(df):
    """
    The same adjusted visibility model on a 24-level hour grid rather than four
    blocks, to confirm that the null is not an artefact of where the block
    boundaries were drawn. Hours with fewer than 10 posts are pooled.
    """
    d = df.copy()
    counts = d["hour"].value_counts()
    d["hour_grp"] = d["hour"].where(d["hour"].isin(counts[counts >= 10].index),
                                    -1).astype(str)
    m = smf.ols(f"log_views ~ C(hour_grp) + C(month) + {CONTENT_MIX}",
                data=d).fit(cov_type="HC3")
    terms = [t for t in m.params.index if t.startswith("C(hour_grp)")]
    R = np.zeros((len(terms), len(m.params)))
    for i, t in enumerate(terms):
        R[i, list(m.params.index).index(t)] = 1
    F = m.f_test(R)
    print(f"[hour grid] joint F = {float(np.squeeze(F.fvalue)):.3f}, "
          f"p = {float(np.squeeze(F.pvalue)):.4f}, N = {int(m.nobs)}")
    return m, F


# ------------------------------------------------------------------- MAIN ----
if __name__ == "__main__":
    df = load()
    df.to_pickle(f"{OUT_DIR}/rq4.pkl")      # reused by the figure script
    print(f"Posts analysed: {len(df)}   "
          f"window: {df.date.min().date()} to {df.date.max().date()}")
    print(f"Total consultation inquiries: {int(df.inquiries.sum())}\n")

    print("--- 1. Descriptives ---")
    descriptives(df)

    print("--- 2. Visibility: time of day ---")
    visibility_models(df, "period")

    print("\n--- 2b. Visibility: weekday ---")
    visibility_models(df, "weekday")

    print("\n--- 3. Inquiry generation: time of day ---")
    inquiry_models(df, "period")

    print("\n--- 3b. Inquiry generation: weekday ---")
    inquiry_models(df, "weekday")

    print("\n--- 4. Era and content-mix diagnostics ---")
    era_and_mix_diagnostics(df)

    print("\n--- 5. Robustness: hour-of-day grid ---")
    hour_robustness(df)

    print("\nDone. Tables written to", OUT_DIR)
