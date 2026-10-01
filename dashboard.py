"""Streamlit dashboard for the predicted cash-out risk map.

Run:
    streamlit run dashboard.py
"""
from pathlib import Path
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.common import load_data
from src.predict import forecast

st.set_page_config(
    page_title="Cash-Out Risk Command Center",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def available_forecast_dates() -> list[pd.Timestamp]:
    dates = []
    pattern = re.compile(r"forecast_(\d{4}-\d{2}-\d{2})\.csv$")
    for path in sorted(C.OUT_DIR.glob("forecast_*.csv")):
        match = pattern.search(path.name)
        if match:
            dates.append(pd.Timestamp(match.group(1)))
    if dates:
        return dates
    return [pd.Timestamp(C.START_DATE) + pd.Timedelta(days=C.N_DAYS)]


@st.cache_data(show_spinner=False)
def load_events_and_cells():
    return load_data()


@st.cache_data(show_spinner="Scoring all ATM clusters…")
def get_full_forecast(date: pd.Timestamp) -> pd.DataFrame:
    events, cells = load_events_and_cells()
    # All cells are returned so the map shows a true risk surface, not only the top-10 table.
    return forecast(date, top_k=len(cells))


def risk_color(level: str) -> str:
    return {"HIGH": "#e53935", "MEDIUM": "#fb8c00", "LOW": "#43a047"}.get(level, "#78909c")


def build_map(df: pd.DataFrame):
    heat = go.Densitymap(
        lat=df["lat"],
        lon=df["lon"],
        z=df["risk_score"],
        radius=35,
        colorscale="YlOrRd",
        zmin=0,
        zmax=1,
        hovertemplate="%{z:.3f}<extra>Predicted risk</extra>",
        name="Risk density",
    )
    alerts = df[df["risk_level"] == "HIGH"].head(15)
    points = go.Scattermap(
        lat=alerts["lat"],
        lon=alerts["lon"],
        mode="markers+text",
        text=alerts.apply(lambda r: f"#{int(r['rank'])} {r['city']}", axis=1),
        textposition="top center",
        marker=dict(size=11, color=[risk_color(x) for x in alerts["risk_level"]], opacity=0.9),
        customdata=alerts[["risk_score", "risk_level", "fraud_type", "r7"]].to_numpy(),
        hovertemplate=(
            "<b>%{text}</b><br>Risk: %{customdata[0]:.3f}<br>"
            "Level: %{customdata[1]}<br>Fraud type: %{customdata[2]}<br>"
            "Recent complaints (7d): %{customdata[3]}<extra></extra>"
        ),
        name="Priority alerts",
    )
    fig = go.Figure([heat, points])
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=22.6, lon=79.0),
            zoom=4.3,
        ),
        height=650,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


st.markdown(
    """
    <div style='padding: 0.3rem 0 1rem 0;'>
      <h1 style='margin-bottom:0;'>🚨 Cash-Out Risk Command Center</h1>
      <p style='font-size:1.05rem;margin-top:0.3rem;'>
        Predictive intelligence for likely cyber-fraud cash-withdrawal hotspots.
        The dashboard prioritises places for human investigation; it is not evidence of wrongdoing.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Forecast controls")
    dates = available_forecast_dates()
    selected_date = st.date_input(
        "Forecast date",
        value=dates[-1].date(),
        min_value=(pd.Timestamp(C.START_DATE) + pd.Timedelta(days=C.WARMUP_DAYS)).date(),
        max_value=(pd.Timestamp(C.START_DATE) + pd.Timedelta(days=C.N_DAYS)).date(),
    )
    selected_date = pd.Timestamp(selected_date)

    full = get_full_forecast(selected_date)
    states = ["All states"] + sorted(full["state"].dropna().unique().tolist())
    state = st.selectbox("Location — state", states)

    cities_df = full if state == "All states" else full[full["state"] == state]
    cities = ["All cities"] + sorted(cities_df["city"].dropna().unique().tolist())
    city = st.selectbox("Location — city", cities)

    fraud_types = ["All fraud types"] + sorted(full["fraud_type"].dropna().unique().tolist())
    fraud_type = st.selectbox("Fraud type", fraud_types)

    min_risk = st.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)
    top_k = st.slider("Alert panel size", 5, 20, 10)

filtered = full.copy()
if state != "All states":
    filtered = filtered[filtered["state"] == state]
if city != "All cities":
    filtered = filtered[filtered["city"] == city]
if fraud_type != "All fraud types":
    filtered = filtered[filtered["fraud_type"] == fraud_type]
filtered = filtered[filtered["risk_score"] >= min_risk].copy()

high = int((filtered["risk_level"] == "HIGH").sum())
medium = int((filtered["risk_level"] == "MEDIUM").sum())
avg_risk = float(filtered["risk_score"].mean()) if not filtered.empty else 0.0
priority = float(filtered.iloc[0]["risk_score"]) if not filtered.empty else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Cells in current view", f"{len(filtered):,}")
k2.metric("High-risk alerts", f"{high:,}")
k3.metric("Average predicted risk", f"{avg_risk:.3f}")
k4.metric("Highest score", f"{priority:.3f}")

st.caption(
    f"Forecast issued for **{selected_date.strftime('%d %b %Y')}** · "
    "Risk is derived from complaint activity available before the forecast day."
)

left, right = st.columns([2.1, 1])
with left:
    st.subheader("Predicted risk heatmap")
    if filtered.empty:
        st.warning("No locations match the current filters.")
    else:
        st.plotly_chart(build_map(filtered), use_container_width=True, config={"displayModeBar": False})

with right:
    st.subheader("Live alert queue")
    alerts = filtered.sort_values(["risk_score", "rank"], ascending=[False, True]).head(top_k)
    if alerts.empty:
        st.info("No active alerts.")
    else:
        for _, row in alerts.iterrows():
            icon = "🔴" if row["risk_level"] == "HIGH" else ("🟠" if row["risk_level"] == "MEDIUM" else "🟢")
            st.markdown(
                f"**{icon} #{int(row['rank'])} · {row['city']}, {row['state']}**  "
                f"<br>Risk **{row['risk_score']:.3f}** · {row['fraud_type']}  "
                f"<br>{int(row['r7'])} recent complaints / 7d",
                unsafe_allow_html=True,
            )
            st.divider()

st.subheader("Investigator view")
display = filtered.sort_values("rank").head(25).copy()
display["risk_score"] = display["risk_score"].round(3)
st.dataframe(
    display[
        ["rank", "city", "state", "risk_score", "risk_level", "fraud_type", "r7", "atm_count", "lat", "lon"]
    ],
    use_container_width=True,
    hide_index=True,
)

st.info(
    "Demo note: the repository uses synthetic NCRP-style data because real complaint/financial trace data is not public. "
    "Predictions are prioritisation signals for trained investigators, not proof against a location, bank, or person."
)
