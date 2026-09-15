"""SRQ2 - statistical analysis.
Produces every table used in the report and workbook, saved to tables/*.csv
plus a stats.json of test results.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, json, os, itertools
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(BASE, "tables")
df = pd.read_csv(os.path.join(BASE, "clean.csv"))
meta = json.load(open(os.path.join(BASE, "meta.json")))
EDU, GOAL, GEN, AGEB, DISC = (meta[k] for k in ["EDU_ORDER", "GOAL_ORDER", "GENDER_ORDER", "AGE_ORDER", "DISC_ORDER"])
N = len(df)
out = {"N": N}
rng = np.random.default_rng(2026)


def freq(series, order=None, name="Category"):
    vc = series.value_counts()
    if order:
        vc = vc.reindex([o for o in order if o in vc.index])
    t = pd.DataFrame({name: vc.index, "n": vc.values})
    t["Percent"] = (t["n"] / len(series) * 100).round(1)
    t["Cumulative percent"] = t["Percent"].cumsum().round(1)
    return t


def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum(); r, k = ct.shape
    phi2 = chi2 / n
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1); kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else np.nan


def perm_chi2(ct, iters=20000):
    """Monte-Carlo p-value for r x c tables with sparse cells."""
    obs = stats.chi2_contingency(ct, correction=False)[0]
    rows = np.repeat(np.arange(ct.shape[0]), ct.values.sum(axis=1))
    cols = np.repeat(np.arange(ct.shape[1]), ct.values.sum(axis=0))
    hits = 0
    for _ in range(iters):
        c = rng.permutation(cols)
        tab = np.zeros(ct.shape)
        np.add.at(tab, (rows, c), 1)
        keep = (tab.sum(1) > 0)[:, None] & (tab.sum(0) > 0)[None, :]
        if not keep.all():
            sub = tab[tab.sum(1) > 0][:, tab.sum(0) > 0]
        else:
            sub = tab
        if sub.shape[0] < 2 or sub.shape[1] < 2:
            continue
        if stats.chi2_contingency(sub, correction=False)[0] >= obs - 1e-9:
            hits += 1
    return obs, (hits + 1) / (iters + 1)


def assoc(a, b, ordera=None, orderb=None, label=""):
    ct = pd.crosstab(df[a], df[b])
    if ordera: ct = ct.reindex([o for o in ordera if o in ct.index])
    if orderb: ct = ct.reindex(columns=[o for o in orderb if o in ct.columns])
    chi2, p, dof, exp = stats.chi2_contingency(ct, correction=False)
    obs_chi2, pmc = perm_chi2(ct)
    small = float((exp < 5).mean() * 100)
    return dict(pair=label or f"{a} x {b}", chi2=round(float(chi2), 3), df=int(dof),
                p_asymptotic=round(float(p), 4), p_monte_carlo=round(float(pmc), 4),
                cramers_v=round(cramers_v(ct), 3), cells_expected_lt5_pct=round(small, 1),
                n=int(ct.values.sum())), ct


# ============ Table 1. Sample profile ============
blocks = []
for var, order, lab in [("gender", GEN, "Gender"), ("age_band", AGEB + ["Not recorded"], "Age band (years)"),
                        ("education", EDU, "Highest qualification held"),
                        ("goal", GOAL, "Study goal pursued"),
                        ("discipline", DISC, "Disciplinary cluster"),
                        ("profile_completeness", None, "Profile record status")]:
    f = freq(df[var], order, name="Level")
    f.insert(0, "Characteristic", lab)
    blocks.append(f)
t1 = pd.concat(blocks, ignore_index=True)[["Characteristic", "Level", "n", "Percent"]]
t1.to_csv(f"{T}/T1_sample_profile.csv", index=False)

# ============ Table 2. Age descriptives ============
def desc(s, label):
    s = s.dropna()
    q1, q3 = np.percentile(s, [25, 75])
    ci = stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=stats.sem(s)) if len(s) > 1 else (np.nan, np.nan)
    return dict(Group=label, n=len(s), Mean=round(s.mean(), 2), SD=round(s.std(ddof=1), 2),
                **{"95% CI lower": round(ci[0], 2), "95% CI upper": round(ci[1], 2)},
                Median=round(s.median(), 1), IQR=round(q3 - q1, 1), Min=int(s.min()), Max=int(s.max()),
                Skewness=round(float(stats.skew(s, bias=False)), 2),
                Kurtosis=round(float(stats.kurtosis(s, bias=False)), 2))

rows = [desc(df["age"], "All booked clients")]
for g in GEN:
    rows.append(desc(df.loc[df.gender == g, "age"], f"Gender: {g}"))
for e in EDU:
    sub = df.loc[df.education == e, "age"]
    if len(sub) >= 3:
        rows.append(desc(sub, f"Qualification: {e}"))
for d in DISC:
    sub = df.loc[df.discipline == d, "age"]
    if len(sub) >= 3:
        rows.append(desc(sub, f"Discipline: {d}"))
t2 = pd.DataFrame(rows)
t2.to_csv(f"{T}/T2_age_descriptives.csv", index=False)

sw = stats.shapiro(df["age"].dropna())
mw = stats.mannwhitneyu(df.loc[df.gender == "Female", "age"], df.loc[df.gender == "Male", "age"])
nF, nM = (df.gender == "Female").sum(), (df.gender == "Male").sum()
rb = 1 - 2 * mw.statistic / (nF * nM)
groups_e = [g["age"].values for _, g in df[df.education != "Not recorded"].groupby("education") if len(g) >= 3]
kw_e = stats.kruskal(*groups_e)
eps2_e = (kw_e.statistic - len(groups_e) + 1) / (sum(len(g) for g in groups_e) - len(groups_e))
groups_d = [g["age"].values for _, g in df[df.discipline != "Not recorded"].groupby("discipline") if len(g) >= 3]
kw_d = stats.kruskal(*groups_d)
eps2_d = (kw_d.statistic - len(groups_d) + 1) / (sum(len(g) for g in groups_d) - len(groups_d))
sp = stats.spearmanr(df["age"], df["edu_rank"], nan_policy="omit")

t3 = pd.DataFrame([
    dict(Test="Shapiro-Wilk normality of age (all clients)", Statistic=round(float(sw.statistic), 3),
         df="", p=round(float(sw.pvalue), 4), **{"Effect size": "", "Interpretation": "Age departs from normality, so rank based tests are used throughout"}),
    dict(Test="Mann-Whitney U, age by gender", Statistic=round(float(mw.statistic), 1), df="",
         p=round(float(mw.pvalue), 4), **{"Effect size": f"rank biserial r = {rb:.3f}",
         "Interpretation": "No reliable age difference between female and male bookers"}),
    dict(Test="Kruskal-Wallis, age by qualification held", Statistic=round(float(kw_e.statistic), 3),
         df=int(len(groups_e) - 1), p=round(float(kw_e.pvalue), 4),
         **{"Effect size": f"epsilon squared = {eps2_e:.3f}", "Interpretation": "Age rises with the qualification already held"}),
    dict(Test="Kruskal-Wallis, age by disciplinary cluster", Statistic=round(float(kw_d.statistic), 3),
         df=int(len(groups_d) - 1), p=round(float(kw_d.pvalue), 4),
         **{"Effect size": f"epsilon squared = {eps2_d:.3f}", "Interpretation": "Age is comparable across disciplines"}),
    dict(Test="Spearman correlation, age and qualification rank", Statistic=round(float(sp.statistic), 3),
         df="", p=round(float(sp.pvalue), 4), **{"Effect size": f"rho = {sp.statistic:.3f}",
         "Interpretation": "Moderate positive monotonic association"}),
])
t3.to_csv(f"{T}/T3_inferential_age.csv", index=False)

# ============ Table 4. Qualification x goal transition matrix ============
sub = df[(df.education != "Not recorded") & (df.goal != "Not recorded")]
ct_eg = pd.crosstab(sub["education"], sub["goal"]).reindex(
    [e for e in EDU if e != "Not recorded"]).reindex(columns=[g for g in GOAL if g != "Not recorded"]).fillna(0).astype(int)
row_pct = (ct_eg.div(ct_eg.sum(1), axis=0) * 100).round(1)
t4 = ct_eg.copy()
t4.insert(0, "Held qualification", t4.index)
t4["Row total"] = ct_eg.sum(1)
t4.to_csv(f"{T}/T4_qualification_by_goal_counts.csv", index=False)
t4p = row_pct.copy(); t4p.insert(0, "Held qualification", t4p.index)
t4p.to_csv(f"{T}/T4b_qualification_by_goal_rowpct.csv", index=False)
res_eg, _ = assoc("education", "goal", EDU, GOAL, "Qualification held x study goal (recorded profiles)")
# recompute restricted to recorded profiles
chi2, p, dof, exp = stats.chi2_contingency(ct_eg, correction=False)
_, pmc = perm_chi2(ct_eg)
res_eg = dict(pair="Qualification held x study goal", chi2=round(float(chi2), 3), df=int(dof),
              p_asymptotic=round(float(p), 4), p_monte_carlo=round(float(pmc), 4),
              cramers_v=round(cramers_v(ct_eg), 3),
              cells_expected_lt5_pct=round(float((exp < 5).mean() * 100), 1), n=int(ct_eg.values.sum()))

# standardised residuals for the heatmap
resid = pd.DataFrame((ct_eg.values - exp) / np.sqrt(exp * np.outer(1 - ct_eg.sum(1) / ct_eg.values.sum(),
                                                                  1 - ct_eg.sum(0) / ct_eg.values.sum())),
                     index=ct_eg.index, columns=ct_eg.columns).round(2)
rr = resid.copy(); rr.insert(0, "Held qualification", rr.index)
rr.to_csv(f"{T}/T5_adjusted_residuals.csv", index=False)

# ============ Table 6. Gender crosstabs ============
gender_tables = {}
for var, order, lab in [("education", EDU, "Highest qualification held"),
                        ("goal", GOAL, "Study goal pursued"),
                        ("discipline", DISC, "Disciplinary cluster"),
                        ("age_band", AGEB, "Age band (years)")]:
    ct = pd.crosstab(df[var], df["gender"]).reindex([o for o in order if o in df[var].unique()]).fillna(0).astype(int)
    ct = ct.reindex(columns=GEN).fillna(0).astype(int)
    pct = (ct.div(ct.sum(0), axis=1) * 100).round(1)
    tab = pd.DataFrame({"Characteristic": lab, "Level": ct.index,
                        "Female n": ct["Female"], "Female %": pct["Female"],
                        "Male n": ct["Male"], "Male %": pct["Male"],
                        "Total n": ct.sum(1)}).reset_index(drop=True)
    gender_tables[var] = tab
t6 = pd.concat(gender_tables.values(), ignore_index=True)
t6.to_csv(f"{T}/T6_gender_crosstabs.csv", index=False)

# ============ Table 7. Association matrix (Cramer's V) ============
vars_ = [("gender", "Gender"), ("age_band", "Age band"), ("education", "Qualification held"),
         ("goal", "Study goal"), ("discipline", "Disciplinary cluster")]
V = pd.DataFrame(np.eye(len(vars_)), index=[l for _, l in vars_], columns=[l for _, l in vars_])
tests = []
for (a, la), (b, lb) in itertools.combinations(vars_, 2):
    d2 = df[(df[a] != "Not recorded") & (df[b] != "Not recorded")]
    ct = pd.crosstab(d2[a], d2[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        continue
    v = cramers_v(ct)
    V.loc[la, lb] = V.loc[lb, la] = round(v, 3)
    chi2, p, dof, exp = stats.chi2_contingency(ct, correction=False)
    _, pmc = perm_chi2(ct, iters=10000)
    tests.append(dict(**{"Variable pair": f"{la} x {lb}", "n": int(ct.values.sum()),
                         "Chi-square": round(float(chi2), 3), "df": int(dof),
                         "p (asymptotic)": round(float(p), 4), "p (Monte Carlo)": round(float(pmc), 4),
                         "Cramer's V (corrected)": round(v, 3),
                         "Expected cells < 5 (%)": round(float((exp < 5).mean() * 100), 1)}))
Vout = V.copy(); Vout.insert(0, "Variable", Vout.index)
Vout.to_csv(f"{T}/T7_association_matrix.csv", index=False)
t8 = pd.DataFrame(tests)
t8.to_csv(f"{T}/T8_chi_square_tests.csv", index=False)

# ============ Table 9. Booking series ============
d = df[df["date"].notna()].copy()
d["date"] = pd.to_datetime(d["date"])
monthly = d.groupby([d["date"].dt.to_period("M").astype(str), "gender"]).size().unstack(fill_value=0)
monthly = monthly.reindex(columns=GEN, fill_value=0)
monthly["Total"] = monthly.sum(1)
monthly["Cumulative total"] = monthly["Total"].cumsum()
monthly["Share of all bookings (%)"] = (monthly["Total"] / monthly["Total"].sum() * 100).round(1)
MON = {"2022-12": "Dec 2022", "2023-01": "Jan 2023", "2023-02": "Feb 2023", "2023-03": "Mar 2023", "2023-04": "Apr 2023"}
monthly.index = [MON.get(i, i) for i in monthly.index]
t9 = monthly.copy(); t9.insert(0, "Month", t9.index)
t9.to_csv(f"{T}/T9_monthly_bookings.csv", index=False)

# monthly composition by qualification, for the stacked figure
mon_edu = d.groupby([d["date"].dt.to_period("M").astype(str), "education"]).size().unstack(fill_value=0)
mon_edu = mon_edu.reindex(columns=[e for e in EDU if e in mon_edu.columns], fill_value=0)
mon_edu.index = [MON.get(i, i) for i in mon_edu.index]
me = mon_edu.copy(); me.insert(0, "Month", me.index)
me.to_csv(f"{T}/T10_monthly_by_qualification.csv", index=False)

# weekly series
wk = d.set_index("date").resample("W-MON").size()
wkt = pd.DataFrame({"Week commencing": wk.index.strftime("%d %b %Y"), "Bookings": wk.values})
wkt["Cumulative"] = wkt["Bookings"].cumsum()
wkt["4 week rolling mean"] = wkt["Bookings"].rolling(4, min_periods=1).mean().round(2)
wkt.to_csv(f"{T}/T11_weekly_bookings.csv", index=False)

# ============ Table 12. Field of study detail ============
t12 = freq(df["field"], name="Field of study")
t12["Disciplinary cluster"] = t12["Field of study"].map(
    {f: dsc for dsc, fs in {k: v for k, v in zip(DISC[:-1], [
        ["Computer Engineering", "Industrial Engineering", "Biomedical Engineering", "Technical Engineering",
         "Mechanical Engineering", "Environment Engineering", "Architecture Engineering", "Production Engineering",
         "Network Engineering", "IT Engineering", "Engineering", "Architecture"],
        ["Chemistry", "Microbiology", "Biotechnology", "Geology", "Biostatistics"],
        ["Medicine", "Nursing", "Health", "Clinical Stomatology"],
        ["Business", "Economy", "Management", "Law", "Social Science", "Psychology"]])}.items() for f in fs}
).fillna("Not recorded")
t12 = t12[["Disciplinary cluster", "Field of study", "n", "Percent"]]
t12.to_csv(f"{T}/T12_field_detail.csv", index=False)

# ============ Table 13. Aspiration gap ============
gapd = df[df["aspiration_gap"].notna()]
gap = gapd.groupby("aspiration_gap").size()
GAPLAB = {-1.0: "Targets a level below the one held", -0.5: "Targets same level or one step down",
          0.0: "Targets the same level as held",
          0.5: "Targets same level or one step up", 1.0: "Targets one level up",
          1.5: "Targets one to two levels up", 2.0: "Targets two levels up"}
t13 = pd.DataFrame({"Aspiration step": [GAPLAB.get(k, str(k)) for k in gap.index],
                    "Numeric step": gap.index, "n": gap.values})
t13["Percent of profiled clients"] = (t13["n"] / t13["n"].sum() * 100).round(1)
t13.to_csv(f"{T}/T13_aspiration_gap.csv", index=False)

# gap by gender for a grouped chart
gapg = pd.crosstab(gapd["aspiration_gap"].map(lambda k: GAPLAB.get(k, str(k))), gapd["gender"]).reindex(columns=GEN, fill_value=0)
gg = gapg.copy(); gg.insert(0, "Aspiration step", gg.index)
gg.to_csv(f"{T}/T14_aspiration_gap_by_gender.csv", index=False)

# ============ Table 15. Age band x qualification matrix (heatmap source) ============
ab_edu = pd.crosstab(df["age_band"], df["education"]).reindex([a for a in AGEB]).reindex(
    columns=[e for e in EDU]).fillna(0).astype(int)
ae = ab_edu.copy(); ae.insert(0, "Age band", ae.index)
ae.to_csv(f"{T}/T15_ageband_by_qualification.csv", index=False)

# discipline x goal
dg = pd.crosstab(df["discipline"], df["goal"]).reindex([x for x in DISC]).reindex(columns=GOAL).fillna(0).astype(int)
dgo = dg.copy(); dgo.insert(0, "Disciplinary cluster", dgo.index)
dgo.to_csv(f"{T}/T16_discipline_by_goal.csv", index=False)

# ============ Table 17. Data completeness ============
comp = []
for col, lab in [("date", "Date of first consultation"), ("gender", "Gender"), ("age", "Age"),
                 ("education", "Highest qualification held"), ("goal", "Study goal"), ("field", "Field of study")]:
    if col in ("date", "age"):
        miss = int(df[col].isna().sum())
    else:
        miss = int((df[col] == "Not recorded").sum())
    comp.append({"Variable": lab, "Recorded n": N - miss, "Recorded %": round((N - miss) / N * 100, 1),
                 "Missing n": miss, "Missing %": round(miss / N * 100, 1)})
t17 = pd.DataFrame(comp)
t17.to_csv(f"{T}/T17_completeness.csv", index=False)

# ============ Correspondence analysis on qualification x goal ============
# the single Diploma -> Bachelor case is a structural outlier that would absorb the
# first dimension on its own, so the map is fitted to the Bachelor-and-above table
ct_ca = ct_eg.drop(index=["Diploma"], errors="ignore").drop(columns=["Bachelor"], errors="ignore")
ct_ca = ct_ca.loc[ct_ca.sum(axis=1) > 0, ct_ca.sum(axis=0) > 0]
P = ct_ca.values / ct_ca.values.sum()
r = P.sum(1); c = P.sum(0)
S = np.diag(1 / np.sqrt(r)) @ (P - np.outer(r, c)) @ np.diag(1 / np.sqrt(c))
U, sv, Vt = np.linalg.svd(S, full_matrices=False)
inertia = sv ** 2
expl = inertia / inertia.sum() * 100
rowc = np.diag(1 / np.sqrt(r)) @ U[:, :2] * sv[:2]
colc = np.diag(1 / np.sqrt(c)) @ Vt.T[:, :2] * sv[:2]
ca = pd.DataFrame(np.vstack([rowc, colc]), columns=["Dimension 1", "Dimension 2"])
ca.insert(0, "Point", list(ct_ca.index) + list(ct_ca.columns))
ca.insert(1, "Point type", ["Qualification held"] * len(ct_ca.index) + ["Study goal"] * len(ct_ca.columns))
ca["Mass"] = np.concatenate([r, c])
ca = ca.round(4)
ca.to_csv(f"{T}/T18_correspondence_coordinates.csv", index=False)
t19 = pd.DataFrame({"Dimension": np.arange(1, len(sv) + 1), "Singular value": sv.round(4),
                    "Principal inertia": inertia.round(4),
                    "Inertia explained (%)": expl.round(1),
                    "Cumulative (%)": expl.cumsum().round(1)})
t19.to_csv(f"{T}/T19_ca_inertia.csv", index=False)

out.update(dict(
    edu_goal=res_eg, chi_tests=tests,
    shapiro=dict(W=float(sw.statistic), p=float(sw.pvalue)),
    mannwhitney=dict(U=float(mw.statistic), p=float(mw.pvalue), rb=float(rb)),
    kruskal_edu=dict(H=float(kw_e.statistic), p=float(kw_e.pvalue), eps2=float(eps2_e), k=len(groups_e)),
    kruskal_disc=dict(H=float(kw_d.statistic), p=float(kw_d.pvalue), eps2=float(eps2_d), k=len(groups_d)),
    spearman=dict(rho=float(sp.statistic), p=float(sp.pvalue)),
    ca_inertia=expl.tolist(), total_inertia=float(inertia.sum()), ca_table=ct_ca.to_dict(),
    age=dict(mean=float(df.age.mean()), sd=float(df.age.std()), median=float(df.age.median()),
             q1=float(df.age.quantile(.25)), q3=float(df.age.quantile(.75)),
             mn=int(df.age.min()), mx=int(df.age.max())),
    gender=df.gender.value_counts().to_dict(),
    education=df.education.value_counts().to_dict(),
    goal=df.goal.value_counts().to_dict(),
    discipline=df.discipline.value_counts().to_dict(),
    age_band=df.age_band.value_counts().to_dict(),
    monthly=monthly["Total"].to_dict(),
))
with open(os.path.join(BASE, "stats.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)

print(json.dumps({k: v for k, v in out.items() if k not in ("chi_tests",)}, indent=1, default=str)[:3000])
print("\nchi tests:")
for t in tests:
    print(t)
print("\nT4 counts:\n", ct_eg)
print("\nresiduals:\n", resid)
print("\nmonthly:\n", monthly)
print("\ngap:\n", t13)
