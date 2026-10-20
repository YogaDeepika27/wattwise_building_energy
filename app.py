from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import ExtraTreesRegressor, IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="WattWise | Building Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

COLORS = {
    "green": "#8BE0A2",
    "blue": "#83B7FF",
    "orange": "#FFBE76",
    "red": "#FF7777",
    "muted": "#9AA8B8",
}


# --------------------------------------------------
# UI STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    :root { --bg:#0A0F16; --panel:#111B26; --panel2:#152331; --line:#263747;
            --text:#EDF4FA; --muted:#9BAEC0; --mint:#91F2B3; --blue:#8ABEFF; }
    .stApp { background:radial-gradient(ellipse at 15% 0%, #132A30 0%, var(--bg) 45%);
             color:var(--text); font-family:'DM Sans',sans-serif; }
    .block-container { width:100% !important; max-width:none !important; padding:1.6rem clamp(1rem, 3vw, 3rem) 3rem !important; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#101B27,#0D151E);
      border-right:1px solid var(--line); }
    [data-testid="stSidebar"] h2 { letter-spacing:-.5px; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,#142331,#101923);
      border:1px solid #2A3C4B; padding:18px 20px; border-radius:18px;
      box-shadow:0 8px 24px rgba(0,0,0,.14); min-height:118px; }
    [data-testid="stMetricLabel"] { color:var(--muted); font-size:.86rem; }
    [data-testid="stMetricValue"] { color:var(--mint); font-weight:800; letter-spacing:-.7px; }
    [data-testid="stMetricDelta"] { font-size:.8rem; }
    .hero { position:relative; overflow:hidden; background:linear-gradient(115deg,rgba(24,66,57,.94),rgba(18,36,54,.96));
      border:1px solid #31564D; padding:30px 32px; border-radius:24px; margin:6px 0 26px;
      box-shadow:0 18px 48px rgba(0,0,0,.18); }
    .hero:after { content:'⚡'; position:absolute; right:36px; top:4px; font-size:100px;
      opacity:.09; transform:rotate(-10deg); }
    .hero h1 { font-size:clamp(28px,3vw,40px); line-height:1.12; letter-spacing:-1.4px;
      margin:0; color:var(--text); font-weight:800; }
    .hero p { color:#C0D4D2; margin:10px 0 0; font-size:1.02rem; max-width:700px; }
    .eyebrow { color:var(--mint); letter-spacing:2.3px; font-size:11px; font-weight:800; margin-bottom:12px; }
    h2, h3 { letter-spacing:-.5px; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line); border-radius:16px; }
    .stButton button, .stDownloadButton button { border-radius:12px; min-height:44px;
      font-weight:700; border:1px solid #365044; transition:all .18s ease; }
    .stButton button[kind="primary"] { background:linear-gradient(110deg,#91F2B3,#5CCF9A);
      color:#092016; border:0; }
    .stButton button:hover, .stDownloadButton button:hover { transform:translateY(-1px); border-color:var(--mint); }
    .stSelectbox, .stSlider, .stNumberInput, .stFileUploader { border-radius:12px; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
    [data-testid="stPlotlyChart"] { background:rgba(17,27,38,.45); border:1px solid #203140;
      padding:10px; border-radius:16px; width:100%; }
    [data-testid="stPlotlyChart"] iframe { width:100% !important; }
    .mini-label { color:var(--muted); font-size:.82rem; letter-spacing:.2px; }
    .forecast-note { background:rgba(20,35,49,.72); border:1px solid #263747;
      border-radius:12px; padding:12px 15px; color:#C6D4E0; font-size:.9rem; }
    [data-testid="stAlert"] { border-radius:12px; }
    .section-note { color:var(--muted); font-size:.9rem; margin-top:-8px; }
    @media (max-width: 768px) {
      .block-container { padding-left:1rem; padding-right:1rem; }
      .hero { padding:23px 20px; }
      .hero:after { right:12px; font-size:70px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DEMO DATA
# --------------------------------------------------
@st.cache_data
def create_demo_data():
    rng = np.random.default_rng(42)
    periods = 24 * 180
    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("h"), periods=periods, freq="h"
    )
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()

    occupancy = np.where(
        (weekday < 5) & (hour >= 8) & (hour <= 18), 1.0, 0.35
    )
    temperature = (
        25 + 7 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        + rng.normal(0, 2, periods)
    )
    energy = (
        32 + 25 * occupancy
        + 1.5 * np.maximum(temperature - 22, 0)
        + 10 * ((hour >= 17) & (hour <= 20))
        + 4 * np.sin(2 * np.pi * hour / 24)
        + rng.normal(0, 4, periods)
    )
    energy = np.maximum(energy, 5)
    for index in [900, 1800, 3000]:
        energy[index:index + 3] *= 2.1

    return pd.DataFrame({
        "timestamp": timestamps,
        "energy_kwh": energy,
        "temperature": temperature,
    })


# --------------------------------------------------
# DATA LOADING AND NORMALIZATION
# --------------------------------------------------
def normalize_data(raw):
    df = raw.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_") for c in df.columns
    ]

    time_candidates = ["timestamp", "datetime", "date", "time"]
    energy_candidates = [
        "energy_kwh", "kwh", "meter_reading", "electricity",
        "appliances", "global_active_power",
    ]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    energy_col = next((c for c in energy_candidates if c in df.columns), None)

    if time_col is None or energy_col is None:
        raise ValueError(
            "CSV must contain a date/time column and an energy column. "
            "Supported energy columns: energy_kwh, kwh, meter_reading, "
            "electricity, appliances, global_active_power."
        )

    # The UCI Appliances file uses month/day/year date strings.
    # format='mixed' handles consistent or mixed date strings in recent pandas.
    try:
        df["timestamp"] = pd.to_datetime(
            df[time_col], errors="coerce", format="mixed", dayfirst=False
        )
    except (TypeError, ValueError):
        df["timestamp"] = pd.to_datetime(
            df[time_col], errors="coerce", dayfirst=False
        )

    df["reading"] = pd.to_numeric(df[energy_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "reading"]).sort_values("timestamp")

    if df.empty:
        raise ValueError("No valid date/time and numeric energy records were found.")

    # UCI Appliances: energy in Wh per 10-minute interval.
    # Sum the Wh values for each hour, then convert hourly Wh to kWh.
    if energy_col == "appliances":
        hourly = (
            df.set_index("timestamp")["reading"]
            .resample("h").sum(min_count=1) / 1000.0
        )
    # UCI household_power_consumption Global_active_power is in kW
    # sampled each minute; sum kW-minutes / 60 to estimate hourly kWh.
    elif energy_col == "global_active_power":
        hourly = (
            df.set_index("timestamp")["reading"]
            .resample("min").mean()
            .resample("h").sum(min_count=1) / 60.0
        )
    else:
        # For generic data, assume each record is an energy amount in kWh.
        # If records are meter readings/cumulative readings, preprocessing
        # must be adapted to calculate differences instead.
        hourly = (
            df.set_index("timestamp")["reading"]
            .resample("h").sum(min_count=1)
        )

    result = hourly.rename("energy_kwh").dropna().reset_index()
    result = result[result["energy_kwh"] >= 0].sort_values("timestamp")

    if len(result) < 169:
        raise ValueError(
            f"Only {len(result)} hourly records remain. At least 169 are "
            "needed for the 168-hour lag feature. Check the date column "
            "and source data."
        )
    return result.reset_index(drop=True)


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
def make_features(data):
    df = data.copy().sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"]

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    for lag in [1, 2, 24, 48, 168]:
        df[f"lag_{lag}"] = df["energy_kwh"].shift(lag)

    df["rolling_24"] = df["energy_kwh"].shift(1).rolling(24).mean()
    df["rolling_168"] = df["energy_kwh"].shift(1).rolling(168).mean()

    features = [
        "hour", "day_of_week", "month", "is_weekend",
        "lag_1", "lag_2", "lag_24", "lag_48", "lag_168",
        "rolling_24", "rolling_168",
    ]
    return df.dropna(subset=features), features


# --------------------------------------------------
# MODEL TRAINING AND TIME-BASED EVALUATION
# --------------------------------------------------
@st.cache_data
def train_model(data):
    featured, features = make_features(data)
    if len(featured) < 100:
        raise ValueError("Not enough usable hourly records to train the model.")
    split = int(len(featured) * 0.8)
    train_all, test = featured.iloc[:split], featured.iloc[split:]
    inner_split = max(int(len(train_all) * 0.8), 1)
    fit_part, validation = train_all.iloc[:inner_split], train_all.iloc[inner_split:]
    candidates = {
        "Random Forest": RandomForestRegressor(n_estimators=220, max_depth=18,
            min_samples_leaf=2, max_features=0.9, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=220, max_depth=18,
            min_samples_leaf=2, max_features=0.9, random_state=42, n_jobs=-1),
    }
    validation_mae = {}
    for name, candidate in candidates.items():
        candidate.fit(fit_part[features], fit_part["energy_kwh"])
        pred = np.maximum(candidate.predict(validation[features]), 0)
        validation_mae[name] = mean_absolute_error(validation["energy_kwh"], pred)
    selected_name = min(validation_mae, key=validation_mae.get)
    model = candidates[selected_name]
    model.fit(train_all[features], train_all["energy_kwh"])
    pred = np.maximum(model.predict(test[features]), 0)
    baseline = np.repeat(float(train_all["energy_kwh"].mean()), len(test))
    metrics = {
        "MAE": mean_absolute_error(test["energy_kwh"], pred),
        "RMSE": np.sqrt(mean_squared_error(test["energy_kwh"], pred)),
        "R2": r2_score(test["energy_kwh"], pred),
        "Baseline MAE": mean_absolute_error(test["energy_kwh"], baseline),
        "Selected model": selected_name,
        "Validation MAE - Random Forest": validation_mae["Random Forest"],
        "Validation MAE - Extra Trees": validation_mae["Extra Trees"],
    }
    model.fit(featured[features], featured["energy_kwh"])
    return model, features, metrics, len(train_all), len(test)


def forecast_next_hours(data, model, features, horizon):
    history = data.sort_values("timestamp").copy().reset_index(drop=True)
    if len(history) < 168:
        raise ValueError("At least 168 hourly readings are needed to forecast.")

    results = []
    for _ in range(int(horizon)):
        next_ts = history["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        y = history["energy_kwh"]

        values = {
            "hour": next_ts.hour,
            "day_of_week": next_ts.dayofweek,
            "month": next_ts.month,
            "is_weekend": int(next_ts.dayofweek >= 5),
            "lag_1": float(y.iloc[-1]),
            "lag_2": float(y.iloc[-2]),
            "lag_24": float(y.iloc[-24]),
            "lag_48": float(y.iloc[-48]),
            "lag_168": float(y.iloc[-168]),
            "rolling_24": float(y.tail(24).mean()),
            "rolling_168": float(y.tail(168).mean()),
        }
        row = pd.DataFrame([values], columns=features)
        prediction = max(float(model.predict(row)[0]), 0.0)
        results.append({"timestamp": next_ts, "predicted_kwh": prediction})
        history = pd.concat(
            [history, pd.DataFrame({
                "timestamp": [next_ts], "energy_kwh": [prediction]
            })],
            ignore_index=True,
        )
    return pd.DataFrame(results)


def find_anomalies(data):
    result = data.copy().sort_values("timestamp").reset_index(drop=True)
    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["anomaly"] = IsolationForest(contamination=0.02, random_state=42).fit_predict(
        result[["energy_kwh", "hour", "day_of_week"]]) == -1
    median = float(result["energy_kwh"].median())
    q1, q3 = result["energy_kwh"].quantile([0.25, 0.75])
    iqr = max(float(q3-q1), 1e-9)
    result["deviation_from_median_pct"] = np.where(
        median > 0, (result["energy_kwh"]-median)/median*100, np.nan)
    result["review_reason"] = np.where(
        result["energy_kwh"] > q3 + 1.5*iqr, "High consumption relative to dataset",
        np.where(result["energy_kwh"] < max(0, q1-1.5*iqr),
                 "Low consumption relative to dataset",
                 "Unusual consumption/time pattern"))
    result.loc[~result["anomaly"], "review_reason"] = ""
    return result


def data_quality_report(raw):
    frame = raw.copy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    time_col = next((c for c in ["timestamp", "datetime", "date", "time"] if c in frame.columns), None)
    energy_col = next((c for c in ["energy_kwh", "kwh", "meter_reading", "electricity", "appliances", "global_active_power"] if c in frame.columns), None)
    report = {"Source rows": len(frame), "Duplicate rows": int(frame.duplicated().sum()),
              "Missing cells": int(frame.isna().sum().sum()),
              "Time column": time_col or "Not detected", "Energy column": energy_col or "Not detected"}
    if time_col:
        ts = pd.to_datetime(frame[time_col], errors="coerce", format="mixed", dayfirst=False)
        report["Invalid timestamps"] = int(ts.isna().sum())
        report["Duplicate timestamps"] = int(ts.duplicated().sum())
    if energy_col:
        vals = pd.to_numeric(frame[energy_col], errors="coerce")
        report["Invalid energy values"] = int(vals.isna().sum())
        report["Negative energy values"] = int((vals < 0).sum())
    return report


def money(value):
    return f"₹{value:,.2f}"


# --------------------------------------------------
# SIDEBAR AND DATA SOURCE
# --------------------------------------------------
st.sidebar.markdown("## ⚡ WattWise")
st.sidebar.caption("BUILDING ENERGY INTELLIGENCE")
st.sidebar.markdown("`●` **SYSTEM ONLINE**")
st.sidebar.caption("Forecast · Detect · Optimize")
st.sidebar.divider()

page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Forecast", "Anomalies", "Optimization", "Data & Model"],
)

uploaded = st.sidebar.file_uploader(
    "Upload energy CSV",
    type=["csv"],
    help="CSV must contain a date/time column and an energy column.",
)

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded, sep=None, engine="python")
        quality_report = data_quality_report(raw)
        data = normalize_data(raw)
        demo_mode = False
        st.sidebar.success("Uploaded dataset loaded")
    except Exception as exc:
        st.sidebar.error(f"Could not load CSV: {exc}")
        st.stop()
else:
    # Prefer the known project dataset when present.
    preferred = DATA_DIR / "energydata_complete.csv"
    local_files = [preferred] if preferred.exists() else sorted(DATA_DIR.glob("*.csv"))
    if local_files:
        try:
            raw = pd.read_csv(local_files[0], sep=None, engine="python")
            quality_report = data_quality_report(raw)
            data = normalize_data(raw)
            demo_mode = False
            st.sidebar.caption(f"Using {local_files[0].name}")
        except Exception as exc:
            st.error(f"Could not process {local_files[0].name}: {exc}")
            st.stop()
    else:
        data = create_demo_data()
        quality_report = {"Source rows": len(data), "Duplicate rows": 0, "Missing cells": 0, "Invalid timestamps": 0, "Invalid energy values": 0}
        demo_mode = True
        st.sidebar.info("Demo data active")

st.sidebar.divider()
electricity_rate = st.sidebar.number_input(
    "Electricity tariff (₹/kWh)",
    min_value=0.0,
    max_value=100.0,
    value=8.0,
    step=0.5,
)
st.sidebar.caption(
    "Forecasts and savings are estimates, not controls for real equipment. Verify source energy units before operational use."
)

with st.spinner("Preparing data and training the energy model..."):
    try:
        model, features, metrics, train_n, test_n = train_model(data)
        anomalies = find_anomalies(data)
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()


# --------------------------------------------------
# COMMON HEADER AND SUMMARY VALUES
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">SMART BUILDING · ENERGY ANALYTICS</div>
        <h1>Energy Intelligence Center</h1>
        <p>Understand consumption. Forecast demand. Find opportunities to save.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if demo_mode:
    st.warning("DEMO MODE — simulated data is being used.")
else:
    st.success(
        f"REAL DATA MODE — {len(data):,} hourly records loaded from your CSV."
    )

latest = data["timestamp"].max()
total_kwh = data["energy_kwh"].sum()
average_kwh = data["energy_kwh"].mean()
peak_kwh = data["energy_kwh"].max()
estimated_cost = total_kwh * electricity_rate


# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------
if page == "Overview":
    st.subheader("Building performance")
    st.caption("A clear snapshot of energy use, operating patterns, and estimated cost.")
    st.caption(
        f"Reporting period: {data['timestamp'].min():%d %b %Y} "
        f"— {latest:%d %b %Y}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total energy", f"{total_kwh:,.1f} kWh")
    c2.metric("Average hourly usage", f"{average_kwh:,.2f} kWh")
    c3.metric("Peak hourly usage", f"{peak_kwh:,.2f} kWh")
    c4.metric("Estimated energy cost", money(estimated_cost))

    st.markdown("### Consumption trends")
    daily = (
        data.set_index("timestamp")["energy_kwh"]
        .resample("D").sum().reset_index()
    )
    left, right = st.columns([1.6, 1])

    with left:
        fig = px.area(daily, x="timestamp", y="energy_kwh",
                      title="Daily energy consumption")
        fig.update_traces(line_color=COLORS["green"],
                          fillcolor="rgba(139,224,162,0.14)")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Date",
            yaxis_title="Energy (kWh/day)",
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        hourly = data.assign(
            hour=data["timestamp"].dt.hour
        ).groupby("hour", as_index=False)["energy_kwh"].mean()
        fig = px.bar(hourly, x="hour", y="energy_kwh",
                     title="Average usage by hour")
        fig.update_traces(marker_color=COLORS["blue"])
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Hour of day",
            yaxis_title="Average kWh",
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    weekly = data.set_index("timestamp")["energy_kwh"].resample("W").sum().reset_index()
    weekly_fig = px.bar(weekly, x="timestamp", y="energy_kwh", title="Weekly energy consumption")
    weekly_fig.update_traces(marker_color=COLORS["orange"])
    weekly_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Week ending",
        yaxis_title="Energy (kWh/week)", height=340)
    st.plotly_chart(weekly_fig, use_container_width=True)

    st.markdown("### Latest meter records")
    st.dataframe(
        data.tail(12).sort_values("timestamp", ascending=False),
        use_container_width=True, hide_index=True,
    )


# --------------------------------------------------
# FORECAST PAGE
# --------------------------------------------------
elif page == "Forecast":
    st.subheader("Energy demand forecast")
    st.write(
        "Predict future hourly usage with a Random Forest model "
        "and recursively generated lag features."
    )
    horizon = st.slider("Forecast horizon (hours)", 6, 72, 24, step=6)

    if st.button("Generate forecast", type="primary"):
        try:
            with st.spinner("Generating forecast..."):
                st.session_state["forecast"] = forecast_next_hours(
                    data, model, features, horizon
                )
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")

    if "forecast" in st.session_state:
        forecast = st.session_state["forecast"]
        fig = go.Figure()
        recent = data.tail(72)
        fig.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["energy_kwh"],
            name="Historical usage",
            line=dict(color=COLORS["blue"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=forecast["timestamp"], y=forecast["predicted_kwh"],
            name="Predicted usage",
            line=dict(color=COLORS["green"], width=3, dash="dash"),
        ))
        fig.update_layout(
            title=dict(text="Historical vs. predicted hourly energy", x=0.02, xanchor="left"),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Date and time", yaxis_title="Energy (kWh)",
            height=500, autosize=True,
            margin=dict(l=22, r=22, t=70, b=85),
            legend=dict(orientation="h", yanchor="top", y=-0.22,
                        xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        fig.update_xaxes(showgrid=False, automargin=True, tickfont=dict(size=11))
        fig.update_yaxes(showgrid=True, gridcolor="rgba(150,170,190,0.15)",
                         zeroline=False, automargin=True)
        st.plotly_chart(fig, use_container_width=True, config={
            "responsive": True, "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        })

        forecast_total = float(forecast["predicted_kwh"].sum())
        forecast_average = float(forecast["predicted_kwh"].mean())
        forecast_cost = forecast_total * electricity_rate
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Forecast energy", f"{forecast_total:,.2f} kWh")
        fc2.metric("Average per hour", f"{forecast_average:,.2f} kWh")
        fc3.metric("Estimated cost", money(forecast_cost))
        st.caption(
            f"Forecast window: {forecast['timestamp'].min():%d %b %Y, %H:%M} "
            f"to {forecast['timestamp'].max():%d %b %Y, %H:%M} · "
            f"Tariff used: {money(electricity_rate)}/kWh."
        )
        st.info(
            "These are model estimates, not measured future consumption. "
            "Cost assumes the selected tariff applies to every forecast kWh; "
            "actual bills may include fixed charges, taxes, and tariff slabs."
        )
        with st.expander("View hourly forecast data"):
            display_forecast = forecast.copy()
            display_forecast["timestamp"] = display_forecast["timestamp"].dt.strftime("%d %b %Y, %H:%M")
            display_forecast["predicted_kwh"] = display_forecast["predicted_kwh"].round(4)
            st.dataframe(display_forecast, use_container_width=True, hide_index=True)
        st.download_button(
            "Download forecast CSV",
            forecast.to_csv(index=False).encode("utf-8"),
            file_name="energy_forecast.csv",
            mime="text/csv",
        )
    else:
        st.info("Choose a forecast horizon and click Generate forecast.")


# --------------------------------------------------
# ANOMALIES PAGE
# --------------------------------------------------
elif page == "Anomalies":
    st.subheader("Unusual consumption detection")
    st.write(
        "Isolation Forest flags readings that differ from the overall "
        "consumption and time-of-day pattern."
    )
    flagged = anomalies[anomalies["anomaly"]].copy()
    a1, a2 = st.columns(2)
    a1.metric("Flagged observations", f"{len(flagged):,}")
    a2.metric("Flagged share", f"{100 * len(flagged) / len(anomalies):.2f}%")

    fig = px.scatter(
        anomalies, x="timestamp", y="energy_kwh", color="anomaly",
        color_discrete_map={True: COLORS["red"], False: COLORS["green"]},
        title="Consumption readings and flagged anomalies",
        labels={"energy_kwh": "Energy (kWh)", "timestamp": "Time"},
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "An anomaly is a review candidate, not proof of a fault. "
        "Occupancy changes, weather and meter errors can also explain spikes."
    )
    if not flagged.empty:
        cols = ["timestamp", "energy_kwh", "deviation_from_median_pct", "review_reason", "hour", "day_of_week"]
        st.dataframe(flagged.sort_values("energy_kwh", ascending=False)[cols],
                     use_container_width=True, hide_index=True)
    else:
        st.success("No observations were flagged. This does not prove all readings are normal.")
    st.download_button(
        "Download flagged readings",
        flagged.to_csv(index=False).encode("utf-8"),
        "energy_anomalies.csv", "text/csv",
    )


# --------------------------------------------------
# OPTIMIZATION PAGE
# --------------------------------------------------
elif page == "Optimization":
    st.subheader("Energy optimization planner")
    st.caption("Scenario modelling only — adjust assumptions to explore possible savings.")
    st.write(
        "Explore potential savings by applying a hypothetical reduction "
        "to a selected share of consumption."
    )
    reduction = st.slider(
        "Assumed consumption reduction (%)", 1, 25, 10
    )
    eligible_share = st.slider(
        "Share of usage affected (%)", 10, 100, 40
    )

    savings_kwh = total_kwh * (eligible_share / 100) * (reduction / 100)
    savings_cost = savings_kwh * electricity_rate
    o1, o2, o3 = st.columns(3)
    o1.metric("Baseline consumption", f"{total_kwh:,.1f} kWh")
    o2.metric("Scenario savings", f"{savings_kwh:,.2f} kWh")
    o3.metric("Estimated cost reduction", money(savings_cost))
    st.progress(reduction / 25)

    st.markdown("### Suggested actions")
    recommendations = [
        ("HVAC scheduling",
         "Review heating and cooling schedules against occupancy hours. "
         "Check comfort and indoor-air requirements before changing settings."),
        ("Peak-demand review",
         "Inspect high-usage hours and consider shifting flexible loads "
         "where tariffs and operations allow."),
        ("Lighting efficiency",
         "Review lighting schedules, occupancy sensors and LED upgrades."),
        ("Anomaly investigation",
         f"Review {int(anomalies['anomaly'].sum())} flagged readings for "
         "equipment issues, unusual operating hours or meter errors."),
    ]
    hourly_profile = data.assign(hour=data["timestamp"].dt.hour).groupby("hour")["energy_kwh"].mean()
    high_hours = sorted(int(h) for h in hourly_profile.nlargest(3).index)
    recommendations.append(("Review high-use hours",
        "Highest average usage hours in this dataset: " + ", ".join(f"{h:02d}:00" for h in high_hours) +
        ". Check occupancy and equipment schedules before changing operations."))
    for title, description in recommendations:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)
    st.caption(
        "Scenario results are hypothetical, not measured savings or guaranteed outcomes."
    )


# --------------------------------------------------
# DATA AND MODEL PAGE
# --------------------------------------------------
elif page == "Data & Model":
    st.subheader("Dataset and model diagnostics")
    d1, d2, d3 = st.columns(3)
    d1.metric("Hourly records", f"{len(data):,}")
    d2.metric("Training samples", f"{train_n:,}")
    d3.metric("Test samples", f"{test_n:,}")

    st.markdown("### Model evaluation")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:.3f} kWh")
    m2.metric("RMSE", f"{metrics['RMSE']:.3f} kWh")
    m3.metric("R² score", f"{metrics['R2']:.3f}")
    st.info(f"Selected model: {metrics['Selected model']}. Random Forest and Extra Trees are compared on validation data from the training period; the latest 20% is held out for chronological testing.")
    b1, b2 = st.columns(2)
    b1.metric("Baseline test MAE", f"{metrics['Baseline MAE']:.3f} kWh")
    b2.metric("Best validation MAE", f"{min(metrics['Validation MAE - Random Forest'], metrics['Validation MAE - Extra Trees']):.3f} kWh")
    st.dataframe(pd.DataFrame([
        {"Model": "Random Forest", "Validation MAE (kWh)": metrics["Validation MAE - Random Forest"]},
        {"Model": "Extra Trees", "Validation MAE (kWh)": metrics["Validation MAE - Extra Trees"]},
    ]), use_container_width=True, hide_index=True)
    st.caption("Lower MAE is better. Compare test MAE against the baseline before claiming an improvement. The selected model is refitted on all history only after test evaluation.")

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)
    fig = px.bar(
        importance, x="importance", y="feature", orientation="h",
        title="Selected model feature importance",
    )
    fig.update_traces(marker_color=COLORS["green"])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(height=max(420, 32 * len(importance)))
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displaylogo": False})

    st.markdown("### Data quality checks")
    st.dataframe(pd.DataFrame([quality_report]).T.rename(columns={0: "Result"}),
                 use_container_width=True)
    st.caption("Quality checks describe the source file; they do not validate its energy units.")

    st.markdown("### Data preview")
    st.dataframe(data.head(30), use_container_width=True, hide_index=True)
    st.download_button(
        "Export normalized dataset",
        data.to_csv(index=False).encode("utf-8"),
        "normalized_energy_data.csv", "text/csv",
    )

st.sidebar.divider()
st.sidebar.caption("WattWise · Energy intelligence prototype")
