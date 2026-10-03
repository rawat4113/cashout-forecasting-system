"""Live Operations: real-time view of the streaming platform (needs the API running).

    uvicorn api:app            # terminal 1
    streamlit run dashboard.py # terminal 2  -> open "Live Operations" in the sidebar
"""
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Live Operations", page_icon="📡", layout="wide")
st.title("📡 Live Operations")
st.caption("Complaint events stream in, the online feature store updates, clusters are re-scored, and every official "
           "forecast is written to a tamper-evident audit chain. Synthetic NCRP-style data.")


def call(method, path, **kw):
    try:
        r = requests.request(method, API + path, timeout=10, **kw)
        r.raise_for_status()
        return r.json()
    except Exception as exc:                       # noqa: BLE001  (UI boundary)
        st.error(f"API not reachable at {API}: {exc}")
        st.stop()


with st.sidebar:
    st.header("Replay (stands in for the live feed)")
    start = st.date_input("Start day", value=pd.Timestamp("2025-06-01").date(),
                          min_value=pd.Timestamp("2024-03-01").date(), max_value=pd.Timestamp("2025-06-01").date())
    days = st.slider("Days to replay", 1, 14, 5)
    speed = st.select_slider("Speed (real seconds per simulated hour)", options=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0], value=0.25)
    c1, c2 = st.columns(2)
    if c1.button("▶ Start", use_container_width=True):
        try:
            call("POST", "/api/v1/demo/replay/start", json=dict(start_date=str(start), days=days, seconds_per_sim_hour=speed))
        except Exception:
            pass
    if c2.button("■ Stop", use_container_width=True):
        call("POST", "/api/v1/demo/replay/stop")
    refresh = st.toggle("Auto-refresh (2 s)", value=True)


def render():
    stats = call("GET", "/api/v1/live/stats")
    status = call("GET", "/api/v1/demo/replay/status")
    alerts = call("GET", "/api/v1/live/alerts", params=dict(limit=12))
    spots = call("GET", "/api/v1/live/hotspots")

    k = st.columns(5)
    k[0].metric("Replay", "● running" if status.get("running") else "○ idle", status.get("sim_time", "")[:16] if status.get("sim_time") else "")
    k[1].metric("Events ingested", f"{stats['events_ingested']:,}", f"{stats['events_rejected']} rejected")
    sl = stats["score_latency"]
    k[2].metric("Re-score all clusters (p50)", f"{sl.get('p50_ms', 0):.1f} ms", f"p99 {sl.get('p99_ms', 0):.1f} ms", delta_color="off")
    k[3].metric("Latest forecast", alerts["date"] or "—", alerts["kind"] or "")
    ok = stats["audit"]["ok"]
    k[4].metric("Audit chain", "✅ intact" if ok else "❌ BROKEN", f"{stats['audit']['entries']} entries", delta_color="off")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Risk surface" + (f" — {alerts['kind'].title()} forecast for {alerts['date']}" if alerts["date"] else ""))
        if spots["hotspots"]:
            df = pd.DataFrame(spots["hotspots"])
            fig = go.Figure([
                go.Densitymap(lat=df.lat, lon=df.lon, z=df.risk_score, radius=35, colorscale="YlOrRd", zmin=0, zmax=1, showscale=False),
                go.Scattermap(lat=df[df.risk_level == "HIGH"].lat, lon=df[df.risk_level == "HIGH"].lon, mode="markers",
                              marker=dict(size=10, color="#e53935"), text=df[df.risk_level == "HIGH"].city, name="HIGH"),
            ])
            fig.update_layout(map=dict(style="open-street-map", center=dict(lat=22.6, lon=79.0), zoom=4.1), height=520,
                              margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No forecast yet — press ▶ Start in the sidebar.")
    with right:
        st.subheader("Alert queue")
        if not alerts["alerts"]:
            st.write("No alerts.")
        for a in alerts["alerts"][:8]:
            badge = {"NEW": "🆕", "ESCALATED": "⬆️", "UNCHANGED": ""}[a["status"]]
            icon = "🔴" if a["level"] == "HIGH" else "🟠"
            with st.expander(f"{icon} #{a['rank']} {a['city']} · {a['risk_score']:.2f} {badge}"):
                st.write(a["recommended_action"])
                for r in a["reasons"]:
                    st.markdown(f"- {r}")
                st.caption(f"{a['alert_id']} · {a['fraud_type']} · human review required")

    with st.expander("Audit trail (hash-chained)"):
        if st.button("Re-verify chain now"):
            st.json(call("GET", "/api/v1/audit/verify"))
        tail = call("GET", "/api/v1/audit/tail", params=dict(n=5))
        if tail:
            st.dataframe(pd.DataFrame([dict(seq=t["seq"], time=t["ts"], kind=t["kind"], hash=t["hash"][:16] + "…",
                                            prev=t["prev"][:16] + "…", model=t["model_sha256"][:10]) for t in tail]),
                         hide_index=True, use_container_width=True)
    st.caption(f"Model fingerprint (sha256): `{stats['model_sha256'][:24]}…` · responsible use: alerts prioritise places for "
               "human review, they are not evidence against any person, bank or community.")


if refresh:
    st.fragment(run_every=2)(render)()
else:
    render()
