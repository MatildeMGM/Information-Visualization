import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="NSF Termination Dashboard", layout="wide")

# ---------- data loading ----------
@st.cache_data(show_spinner=False)
def load_data():
    data_dir = Path(__file__).parent / "project1"   # robust on Streamlit Cloud

    # read as strings where it helps merges be stable
    df_nsf   = pd.read_csv(data_dir / "nsf_terminations_airtable.csv", dtype={"grant_id": str})
    df_flags = pd.read_csv(data_dir / "flagged_words_trump_admin.csv", dtype={"grant_id": str})
    df_cruz  = pd.read_csv(data_dir / "cruz_list.csv", dtype={"grant_id": str})

    # merge (left joins keep all grants from nsf table)
    df = (
        df_nsf
        .merge(df_flags, on="grant_id", how="left")
        .merge(df_cruz,  on="grant_id", how="left")
    )

    # ---- light schema guards (avoid KeyErrors if cols are missing) ----
    for col, default in {
        "has_flagged_word": False,
        "in_cruz_list": False,
        "reinstated": 0,
        "status": 1,                 # assume 1=active unless stated 0
        "nsf_total_budget": np.nan,
        "org_name": "Unknown",
        "org_state": "NA",
    }.items():
        if col not in df.columns:
            df[col] = default

    # types / cleaning
    df["has_flagged_word"] = df["has_flagged_word"].fillna(False).astype(bool)
    df["in_cruz_list"]     = df["in_cruz_list"].fillna(False).astype(bool)
    df["reinstated"]       = pd.to_numeric(df["reinstated"], errors="coerce").fillna(0).astype(int)
    df["status"]           = pd.to_numeric(df["status"], errors="coerce").fillna(1).astype(int)
    df["nsf_total_budget"] = pd.to_numeric(df["nsf_total_budget"], errors="coerce")

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
).properties(width=480, height=560)

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
).properties(width=620, height=300)

# ---------- Q1: states ----------
state_counts = (
    df_merged[df_merged['status'] == 0]['org_state']
    .fillna('NA').value_counts().reset_index()
)
state_counts.columns = ['state', 'Terminations']
top_states = state_counts.head(15)
max_x = int(top_states['Terminations'].max()) if len(top_states) else 0

chart_states = (
    alt.Chart(top_states, title='Top 15 States by NSF Terminations')
    .encode(
        y=alt.Y('state:N', sort='-x', title=None),
        x=alt.X('Terminations:Q', title=None, scale=alt.Scale(domain=[0, max_x + 10]))
    ).mark_rule(color='lightgray')
    + alt.Chart(top_states).mark_circle(color='firebrick').encode(size=alt.Size('Terminations:Q', legend=None))
    + alt.Chart(top_states).mark_text(align='left', dx=10).encode(text='Terminations:Q')
).properties(width=300, height=240)

# ---------- Q5: Cruz list ----------
rates = (
    df_merged.groupby('in_cruz_list', as_index=False)
    .agg(CancelRate=('status', lambda s: (s == 0).mean()),
         ReinstateRate=('reinstated', 'mean'))
)
rates['Cruz'] = rates['in_cruz_list'].map({True: 'In Cruz list', False: 'Not in Cruz list'})

two_part = rates.melt(id_vars='Cruz', value_vars=['CancelRate','ReinstateRate'],
                      var_name='Metric', value_name='Rate')
two_part['Metric'] = two_part['Metric'].map({'CancelRate':'Cancelled','Reinstated':'Reinstated'})

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
left, right = st.columns([1.05, 1.55], gap="medium")
with left:
    st.subheader("Q4 · Flagged words & termination rate")
    st.altair_chart(chart_flag, width="stretch")

with right:
    st.subheader("Q2/Q3 · Institutions most affected")
    st.altair_chart(chart_inst, width="stretch")
    st.divider()
    r1, r2 = st.columns(2, gap="medium")
    with r1:
        st.subheader("Q1 · States")
        st.altair_chart(chart_states, width="stretch")
    with r2:
        st.subheader("Q5 · Cruz list & reinstatements")
        st.altair_chart(chart_cruz, width="stretch")

st.caption("Answers Q1–Q5 on one page. Notebook with methods & extra visuals: `project_new.ipynb`.")
