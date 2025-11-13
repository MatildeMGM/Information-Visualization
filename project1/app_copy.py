
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from vega_datasets import data
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIG + CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="NSF Termination Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)
if st.button("ℹ️ About this dashboard"):
    st.write("**Authors:** Matilde Matell & Steffen Sørensen")
    st.write("This dashboard visualizes NSF terminations across multiple dimensions...")



st.markdown("""
<style>
div.block-container {padding-top: 0.8rem; padding-bottom: 0.6rem; max-width: 1400px;}
h1,h2,h3 {margin-top: .35rem; margin-bottom: .35rem;}
hr {margin: .5rem 0;}
/* keep title visible */
header {visibility: visible;}
#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)



# ---------------------------------------------------------
# COLORS + ALTAIR GLOBAL CONFIG (MATCH NOTEBOOK)
# ---------------------------------------------------------
PALETTE_RED         = "#6CBAC7"   # base teal (used for budgets)
PALETTE_GREY        = "#BFC5CE"   # neutral grey
PALETTE_FLAGGED     = "#2F7F94"   # darker teal for flagged / cancelled focus
PALETTE_CANCELLED   = "#2F7F94"   # cancelled in Cruz chart
PALETTE_REINSTATED  = "#B5D5DB"   # reinstated in Cruz chart

alt.themes.enable("none")
alt.data_transformers.disable_max_rows()

def style_chart(chart: alt.Chart) -> alt.Chart:
    """Apply the same axis/view styling as in the notebook."""
    return (
        chart
        .configure_axis(grid=False, domain=False)
        .configure_view(stroke=None)
    )

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    data_dir = Path(__file__).parent / "project1"

    def read_csv_smart(path):
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path)

    df_nsf   = read_csv_smart(data_dir / "nsf_terminations_airtable.csv")
    df_cruz  = read_csv_smart(data_dir / "cruz_list.csv")
    df_flags = read_csv_smart(data_dir / "flagged_words_trump_admin.csv")

    # normalize names
    norm = lambda s: str(s).strip().lower().replace(" ", "_")
    df_nsf.columns   = [norm(c) for c in df_nsf.columns]
    df_cruz.columns  = [norm(c) for c in df_cruz.columns]
    df_flags.columns = [norm(c) for c in df_flags.columns]

    # keys + types
    if "grant_id" not in df_cruz.columns and "grant_number" in df_cruz.columns:
        df_cruz["grant_id"] = df_cruz["grant_number"].astype(str).str.strip()
    if "grant_id" in df_nsf.columns:
        df_nsf["grant_id"] = df_nsf["grant_id"].astype(str).str.strip()

    status_map = {"❌ Terminated": 0, "🔄 Possibly Reinstated": 1}
    df_nsf["status"] = df_nsf.get("status", 1)
    df_nsf["status"] = pd.Series(df_nsf["status"]).map(status_map).fillna(1).astype(int)

    df_cruz["in_cruz_list"] = (
        df_cruz.get("in_cruz_list", False)
        .astype(str).str.upper().map({"TRUE": True, "1": True, "FALSE": False, "0": False})
        .fillna(False)
    )
    df_nsf["nsf_total_budget"] = pd.to_numeric(df_nsf.get("nsf_total_budget", np.nan), errors="coerce")
    df_nsf["reinstated"] = pd.to_numeric(df_nsf.get("reinstated", 0), errors="coerce").fillna(0).astype(int)

    df = df_nsf.merge(df_cruz[["grant_id", "in_cruz_list"]], on="grant_id", how="left")

    # flagged words
    import re
    flag_col = "flagged_word" if "flagged_word" in df_flags.columns else ("fla" if "fla" in df_flags.columns else None)
    if flag_col:
        words = df_flags[flag_col].dropna().astype(str).str.strip().str.lower().tolist()
        escaped = [re.escape(w) for w in words if w]
        pattern = r'\b(' + '|'.join(escaped) + r')\b' if escaped else r'^\b$'
    else:
        pattern = r'^\b$'
    text = (df.get("project_title", "").fillna("") + " " + df.get("abstract", "").fillna("")).str.lower()
    df["has_flagged_word"] = text.str.contains(pattern, regex=True, na=False)

    return df

df_merged = load_data()

# ---------------------------------------------------------
# PAGE TITLE (single, visible)
# ---------------------------------------------------------
st.markdown("# NSF Terminations Overview")
st.caption("Final visualization dashboard for Information Visualization Project 1 · Steffen Sørensen & Matilde Matell")

# Heights (layout skal bevares, så vi lader disse stå)
FLAG_H = 560     # tall chart on the left (≈ one 15″ screen height)
SMALL_H = 240    # compact height for the two top-right charts
MAP_H = 360      # short map row height

# ---------------------------------------------------------
# CHART A – Flagged words (tall)  [MATCH NOTEBOOK]
# ---------------------------------------------------------
df_rate = (
    df_merged.assign(
        Flagged=np.where(df_merged["has_flagged_word"], "Flagged", "Not flagged"),
        Cancelled=(df_merged["status"] == 0).astype(int)
    )
    .groupby("Flagged", as_index=False)
    .agg(CancelRate=("Cancelled", "mean"))
)

chart_flag = (
    alt.Chart(df_rate)
    .mark_bar(size=60)
    .encode(
        x=alt.X(
            "Flagged:N",
            title=None,
            axis=alt.Axis(labelAngle=0, labelFontSize=12, grid=False),
        ),
        y=alt.Y(
            "CancelRate:Q",
            title="Share of grants terminated",
            axis=alt.Axis(format=".0%", labelFontSize=11, grid=False),
        ),
        color=alt.Color(
            "Flagged:N",
            scale=alt.Scale(
                domain=["Flagged", "Not flagged"],
                range=[PALETTE_FLAGGED, PALETTE_GREY],
            ),
            legend=None,
        ),
        tooltip=[alt.Tooltip("CancelRate:Q", format=".1%")],
    )
    + alt.Chart(df_rate)
    .mark_text(align="center", dy=-8, fontSize=11, color="black")
    .encode(
        x="Flagged:N",
        y="CancelRate:Q",
        text=alt.Text("CancelRate:Q", format=".1%")
    )
).properties(width="container", height=FLAG_H)

chart_flag = style_chart(chart_flag)

# ---------------------------------------------------------
# CHART B – Institutions (Top 10)  [MATCH NOTEBOOK COLORS]
# ---------------------------------------------------------
tmp = df_merged.copy()
tmp["nsf_total_budget"] = pd.to_numeric(tmp["nsf_total_budget"], errors="coerce")

inst_data = (
    tmp[tmp["status"] == 0]
    .groupby("org_name", dropna=False)
    .agg(
        BudgetCancelled=("nsf_total_budget", "sum"),
        CancelledGrants=("nsf_total_budget", "count"),
    )
    .reset_index()
)

inst_data["BudgetMillions"] = inst_data["BudgetCancelled"] / 1e6
top10 = inst_data.sort_values("BudgetMillions", ascending=False).head(10).copy()

xmax = float(top10["BudgetMillions"].max()) if len(top10) else 1.0

base = alt.Chart(top10).encode(
    y=alt.Y(
        "org_name:N",
        sort=top10["org_name"].tolist(),
        title=None,                     # matcher notebook (ingen y-titel)
        axis=alt.Axis(labelLimit=0),
    ),
    x=alt.X(
        "BudgetMillions:Q",
        title="Total Cancelled Budget (M USD)",
        axis=alt.Axis(labelFontSize=11, grid=False),
        scale=alt.Scale(domain=[0, xmax * 1.08]),
    ),
)

bars = base.mark_bar(color=PALETTE_RED).encode(
    tooltip=[
        alt.Tooltip("org_name:N", title="Institution"),
        alt.Tooltip("BudgetMillions:Q", title="Total Cancelled Budget (M USD)", format=".1f"),
        alt.Tooltip("CancelledGrants:Q", title="Number of Terminated Grants"),
    ]
)

labels = (
    base.mark_text(
        align="left",
        dx=4,
        baseline="middle",
        fontSize=11,
        color="black",
    )
    .transform_calculate(
        label_text='format(datum.CancelledGrants, ".0f") + " grants"'
    )
    .encode(text="label_text:N")
)

chart_inst = (bars + labels).properties(
    width=550,
    height=26 * len(top10) + 20,
)

chart_inst = style_chart(chart_inst)

# ---------------------------------------------------------
# CHART C – Cruz list (grouped bars)  [MATCH NOTEBOOK]
# ---------------------------------------------------------
rates = (
    df_merged.groupby("in_cruz_list", as_index=False)
    .agg(
        CancelRate=("status", lambda s: (s == 0).mean()),
        ReinstateRate=("reinstated", "mean"),
    )
)
rates["Cruz"] = rates["in_cruz_list"].map({True: "In Cruz list", False: "Not in Cruz list"})

two_part = rates.melt(
    id_vars="Cruz",
    value_vars=["CancelRate", "ReinstateRate"],
    var_name="Metric",
    value_name="Rate",
)
two_part["Metric"] = two_part["Metric"].map(
    {"CancelRate": "Cancelled", "ReinstateRate": "Reinstated"}
)

chart_cruz = (
    alt.Chart(two_part)
    .mark_bar()
    .encode(
        x=alt.X(
            "Cruz:N",
            title=None,
            sort=["Not in Cruz list", "In Cruz list"],
            axis=alt.Axis(labelAngle=45, labelFontSize=12, grid=False),
        ),
        xOffset="Metric:N",  # side-by-side bars (grouped)
        y=alt.Y(
            "Rate:Q",
            title="Share of grants",
            stack=None,
            axis=alt.Axis(format=".0%", labelFontSize=11, grid=False),
        ),
        color=alt.Color(
            "Metric:N",
            scale=alt.Scale(
                domain=["Cancelled", "Reinstated"],
                range=[PALETTE_CANCELLED, PALETTE_REINSTATED],
            ),
            title=None,
        ),
        tooltip=[
            "Cruz:N",
            "Metric:N",
            alt.Tooltip("Rate:Q", format=".1%"),
        ],
    )
).properties(width="container", height=SMALL_H)

chart_cruz = style_chart(chart_cruz)

# ---------------------------------------------------------
# CHART D – USA map (centered under right two charts) [MATCH NOTEBOOK]
# ---------------------------------------------------------
map_topo = alt.topo_feature(data.us_10m.url, feature="states")

df_q1 = (
    df_merged.loc[df_merged["status"] == 0, ["org_state", "grant_id"]]
    .assign(org_state=lambda d: d["org_state"].astype(str).str.upper().str.strip())
    .groupby("org_state", as_index=False)["grant_id"].nunique()
    .rename(columns={"grant_id": "Terminations"})
)

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
df_q1["state_name"] = df_q1["org_state"].map(abbr2name)
df_q1 = df_q1.dropna(subset=["state_name"])

capitals = data.us_state_capitals.url

us_base = (
    alt.Chart(map_topo)
    .mark_geoshape(fill="lightgray", stroke="white")
    .project("albersUsa")
)

points = (
    alt.Chart(df_q1)
    .transform_lookup(
        lookup="state_name",
        from_=alt.LookupData(capitals, key="state", fields=["state", "lat", "lon"]),
    )
    .transform_filter(alt.datum.lat != None)
    .mark_circle(size=120, opacity=0.9, stroke="black", strokeWidth=0.6)
    .encode(
        longitude="lon:Q",
        latitude="lat:Q",
        color=alt.Color(
            "Terminations:Q",
            title="Terminations",
            scale=alt.Scale(
                range=["#F1F4F7", "#B5D5DB", "#78B8C4", "#2F7F94"]
            ),
        ),
        tooltip=["state_name:N", "org_state:N", "Terminations:Q"],
    )
)

chart_states = (us_base + points).properties(
    width="container",
    height=MAP_H,
)

chart_states = style_chart(chart_states)

# ---------- LAYOUT: left = tall chart, right = two small + centered map ----------
# (Beholder samme forhold og højder som i din nuværende app)

FLAG_H  = 560   # tall left chart
SMALL_H = 230   # small top-right charts
MAP_H   = 320   # map height

chart_flag   = chart_flag.properties(height=FLAG_H,  width="container")
chart_inst   = chart_inst.properties(height=SMALL_H, width="container")
chart_cruz   = chart_cruz.properties(height=SMALL_H, width="container")
chart_states = chart_states.properties(height=MAP_H,  width="container")

# Two main columns
left_col, right_col = st.columns([1.05, 2.35], gap="small")

with left_col:
    st.markdown("**Termination rate by flagged words**")
    st.altair_chart(chart_flag, use_container_width=True)

with right_col:
    # Top-right: two compact charts
    top1, top2 = st.columns([1.7, 1.0], gap="small")
    with top1:
        st.markdown("**Top 10 institutions with highest amount of cancelled budget**")
        st.altair_chart(chart_inst, use_container_width=True)
    with top2:
        st.markdown("**Correlation between cancelled/reinstated grants and Cruz's list**")
        st.altair_chart(chart_cruz, use_container_width=True)

    # Map centered under the two top-right charts
    spacer_left, map_center, spacer_right = st.columns([0.07, 0.86, 0.07], gap="small")
    with map_center:
        st.markdown("**Distribution of terminated grants by state**")
        st.altair_chart(chart_states, use_container_width=True)



