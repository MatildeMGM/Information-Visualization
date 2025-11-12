


'''

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from vega_datasets import data
from pathlib import Path

st.set_page_config(page_title="NSF Termination Dashboard", layout="wide")


# ---------- data loading ----------
@st.cache_data(show_spinner=False)
def load_data():
    data_dir = Path(__file__).parent / "project1"   # tjek at mappen hedder præcis 'project1'

    # --- smart CSV-læser (fanger også ; fra dansk Excel) ---
    def read_csv_smart(path):
        import pandas as pd
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path)

    # --- læs rå data ---
    df_nsf   = read_csv_smart(data_dir / "nsf_terminations_airtable.csv")
    df_cruz  = read_csv_smart(data_dir / "cruz_list.csv")
    df_flags = read_csv_smart(data_dir / "flagged_words_trump_admin.csv")

    # --- normaliser kolonnenavne (lowercase + underscores) ---
    norm = lambda s: str(s).strip().lower().replace(" ", "_")
    df_nsf.columns   = [norm(c) for c in df_nsf.columns]
    df_cruz.columns  = [norm(c) for c in df_cruz.columns]
    df_flags.columns = [norm(c) for c in df_flags.columns]

    # ---------- CRUZ: lav grant_id (string) + bool ----------
    # dine rå data har grant_number + in_cruz_list i separate kolonner
    if "grant_id" not in df_cruz.columns and "grant_number" in df_cruz.columns:
        df_cruz["grant_id"] = df_cruz["grant_number"].astype(str).str.strip()

    if "in_cruz_list" in df_cruz.columns:
        # håndter TRUE/FALSE/1/0 som bool
        df_cruz["in_cruz_list"] = (
            df_cruz["in_cruz_list"]
            .astype(str).str.strip().str.upper()
            .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
            .fillna(False)
            .astype(bool)
        )
    else:
        df_cruz["in_cruz_list"] = False

    # ---------- NSF: sikre typer + status mapping ----------
    # nøglefelt skal være string for stabil merge
    if "grant_id" in df_nsf.columns:
        df_nsf["grant_id"] = df_nsf["grant_id"].astype(str).str.strip()

    status_map = {"❌ Terminated": 0, "🔄 Possibly Reinstated": 1}
    if "status" in df_nsf.columns:
        df_nsf["status"] = (
            df_nsf["status"].map(status_map).fillna(1).astype(int)
        )
    else:
        df_nsf["status"] = 1

    # budget/reinstated til numerisk
    df_nsf["nsf_total_budget"] = pd.to_numeric(df_nsf.get("nsf_total_budget", np.nan), errors="coerce")
    df_nsf["reinstated"]       = pd.to_numeric(df_nsf.get("reinstated", 0), errors="coerce").fillna(0).astype(int)

    # hold kun relevante kolonner, hvis de findes
    keep = [
        "grant_id", "org_name", "org_state", "org_city",
        "project_title", "abstract",
        "nsf_total_budget", "status",
        "termination_date", "reinstated", "reinstatement_date"
    ]
    df_nsf = df_nsf[[c for c in keep if c in df_nsf.columns]].copy()

    # ---------- MERGE: NSF + CRUZ ----------
    df = df_nsf.merge(df_cruz[["grant_id", "in_cruz_list"]], on="grant_id", how="left")

    # ---------- FLAGGED WORDS: brug kolonnen 'fla' (eller 'flagged_word' hvis findes) ----------
    import re
    flag_col = "flagged_word" if "flagged_word" in df_flags.columns else ("fla" if "fla" in df_flags.columns else None)
    if flag_col is not None:
        words = (
            df_flags[flag_col].dropna().astype(str).str.strip().str.lower().tolist()
        )
    else:
        words = []

    escaped = [re.escape(w) for w in words if w]
    pattern = r'\b(' + '|'.join(escaped) + r')\b' if escaped else r'^\b$'  # tom matcher intet

    text = (df.get("project_title", "").fillna("") + " " + df.get("abstract", "").fillna("")).str.lower()
    df["has_flagged_word"] = text.str.contains(pattern, regex=True, na=False)

    # ---------- slut: default-kolonner hvis noget mangler ----------
    for col, default in {
        "has_flagged_word": False,
        "in_cruz_list": False,
        "reinstated": 0,
        "status": 1,
        "nsf_total_budget": np.nan,
        "org_name": "Unknown",
        "org_state": "NA",
    }.items():
        if col not in df.columns:
            df[col] = default

    return df


df_merged = load_data()

st.title("NSF Terminations Overview")
st.caption("Explore where terminations occur, which institutions are most affected, and correlations.")

# ---------- Q4: flagged words ----------
df_rate = (
    df_merged.assign(
        Flagged=np.where(df_merged['has_flagged_word'], 'Flagged', 'Not flagged'),
        Cancelled=(df_merged['status'] == 0).astype(int)
    )
    .groupby('Flagged', as_index=False)
    .agg(CancelRate=('Cancelled', 'mean'))
)

chart_flag = (
    alt.Chart(df_rate, title='Termination Rate by Flagged Words')
    .mark_bar(size=90)
    .encode(
        x=alt.X('Flagged:N', title=None),
        y=alt.Y('CancelRate:Q', title='Share of Grants Terminated', axis=alt.Axis(format='.0%')),
        color=alt.Color('Flagged:N',
                        scale=alt.Scale(domain=['Flagged','Not flagged'], range=['indianred','#B0B0B0']),
                        legend=None),
        tooltip=[alt.Tooltip('CancelRate:Q', format='.1%')]
    )
    + alt.Chart(df_rate).mark_text(align='center', dy=-8, fontSize=12)
      .encode(x='Flagged:N', y='CancelRate:Q', text=alt.Text('CancelRate:Q', format='.1%'))
).properties(width=300, height=560)

# ---------- Q3/Q2: institutions ----------
inst = (
    df_merged[df_merged['status'] == 0]
    .groupby('org_name', dropna=False)
    .agg(BudgetCancelled=('nsf_total_budget', 'sum'),
         CancelledGrants=('nsf_total_budget', 'count'))
    .reset_index()
)
inst['BudgetMillions'] = np.round(inst['BudgetCancelled'] / 1e6)
top15_inst = inst.sort_values('BudgetCancelled', ascending=False).head(15)

chart_inst = (
    alt.Chart(top15_inst, title='Top 15 Institutions by Cancelled Budget')
    .mark_bar(color='indianred')
    .encode(
        y=alt.Y('org_name:N', sort=top15_inst['org_name'].tolist(), title='Institution'),
        x=alt.X('CancelledGrants:Q', title='Number of Cancelled Grants'),
        tooltip=['org_name:N', alt.Tooltip('BudgetCancelled:Q', title='Budget (USD)', format=',')]
    )
    + alt.Chart(top15_inst).mark_text(align='left', dx=6, fontSize=11)
      .transform_calculate(label_text='format(datum.BudgetMillions, ".0f") + "M USD"')
      .encode(text='label_text:N')
).properties(width=20, height=300)

# ---------- Q1: states ----------
# ---------- Q1: USA map med terminations som cirkler ----------
# Basiskort
map = alt.topo_feature(data.us_10m.url, feature='states')
usChart = (
    alt.Chart(map)
    .mark_geoshape(fill='lightgray', stroke='white')
    .properties(width=600, height=360)
    .project('albersUsa')
)

# Aggreger terminations pr. state (unikke grant_id med status==0)
df_q1 = (
    df_merged.loc[df_merged['status'] == 0, ['org_state', 'grant_id']]
    .assign(org_state=lambda d: d['org_state'].astype(str).str.upper().str.strip())
    .groupby('org_state', as_index=False)['grant_id'].nunique()
    .rename(columns={'grant_id': 'Terminations'})
)

# Map fra state-abbrev til fulde navne (inkl. DC)
abbr2name = {
    'AL':'Alabama','AK':'Alaska','AZ':'Arizona','AR':'Arkansas','CA':'California','CO':'Colorado',
    'CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia','HI':'Hawaii','ID':'Idaho',
    'IL':'Illinois','IN':'Indiana','IA':'Iowa','KS':'Kansas','KY':'Kentucky','LA':'Louisiana',
    'ME':'Maine','MD':'Maryland','MA':'Massachusetts','MI':'Michigan','MN':'Minnesota','MS':'Mississippi',
    'MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey',
    'NM':'New Mexico','NY':'New York','NC':'North Carolina','ND':'North Dakota','OH':'Ohio','OK':'Oklahoma',
    'OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota',
    'TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia','WA':'Washington',
    'WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming','DC':'District of Columbia'
}
df_q1['state_name'] = df_q1['org_state'].map(abbr2name)

# Fjern rækker uden match til navn (ukendte koder/NaN)
df_q1 = df_q1.dropna(subset=['state_name'])

# Hent lat/lon for statshovedstæder til at placere cirkler
capitals = data.us_state_capitals.url

terminationsMap = (
    alt.Chart(df_q1, title='Terminated NSF Grants distributed by State')
    .transform_lookup(
        lookup='state_name',
        from_=alt.LookupData(capitals, key='state', fields=['state', 'lat', 'lon'])
    )
    .transform_filter(alt.datum.lat != None)
    .mark_circle(size=120, opacity=0.85, stroke='white', strokeWidth=0.4)
    .encode(
        longitude='lon:Q',
        latitude='lat:Q',
        color=alt.Color('Terminations:Q', title='Terminations', scale=alt.Scale(scheme='reds')),
        tooltip=['state_name:N', 'org_state:N', 'Terminations:Q']
    )
)

chart_states = (usChart + terminationsMap).configure_view(stroke=None)

# ---------- Q5: Cruz list ----------
rates = (
    df_merged.groupby('in_cruz_list', as_index=False)
    .agg(CancelRate=('status', lambda s: (s == 0).mean()),
         ReinstateRate=('reinstated', 'mean'))
)
rates['Cruz'] = rates['in_cruz_list'].map({True: 'In Cruz list', False: 'Not in Cruz list'})

two_part = rates.melt(id_vars='Cruz', value_vars=['CancelRate','ReinstateRate'],
                      var_name='Metric', value_name='Rate')
two_part['Metric'] = two_part['Metric'].map({'CancelRate':'Cancelled','ReinstateRate':'Reinstated'})


chart_cruz = (
    alt.Chart(two_part, title='Cancelled + Reinstated by Cruz List')
    .mark_bar()
    .encode(
        x=alt.X('Cruz:N', title=None, sort=['Not in Cruz list','In Cruz list']),
        y=alt.Y('Rate:Q', axis=alt.Axis(format='.0%'), title='Share of Grants'),
        color=alt.Color('Metric:N',
                        scale=alt.Scale(domain=['Cancelled','Reinstated'], range=['indianred','#B0B0B0']),
                        title=None),
        tooltip=['Cruz:N','Metric:N', alt.Tooltip('Rate:Q', format='.1%')]
    )
).properties(width=300, height=240)

# ---------- Layout ----------

# Øverste række: Q2/Q3 — institutioner
st.subheader("Q2/Q3 · Institutions most affected")
st.altair_chart(chart_inst, use_container_width=True)
st.divider()

# Midterste række: Q4 — flagged words
st.subheader("Q4 · Flagged words & termination rate")
st.altair_chart(chart_flag, use_container_width=True)
st.divider()

# Nederste række: Q5 (Cruz) og Q1 (map) side om side
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.subheader("Q5 · Cruz list & reinstatements")
    st.altair_chart(chart_cruz, use_container_width=True)

with col2:
    st.subheader("Q1 · States")
    st.altair_chart(chart_states, use_container_width=True)

# Fodnote
st.caption("Answers Q1–Q5 on one page. Notebook with methods & extra visuals: `project_new.ipynb`.")
'''

