import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
import numpy as np
import glob
import os
from io import StringIO
import re

# --- Streamlit Configuration ---
st.set_page_config(page_title="Electricity Demand Dashboard", layout="wide")
st.title(f"Electricity Demand Dashboard – {date.today().year}")

with st.expander("How to Use This App"):
    st.markdown("""
    This dashboard allows you to explore electricity demand and generation trends in Spain across multiple days.

    **Steps to Explore the Dashboard:**

    1. **Click the button** below to activate the dashboard.
    2. In the **Trend Visualization** section:
       - Select a specific day from the dropdown.
       - Adjust the time range using the slider to view real vs. forecasted demand.
    3. In **Key Performance Metrics**:
       - Choose which indicators (e.g., volatility, VaR) you want to display.
       - Metrics are calculated based on the selected day's demand.
    4. In **Detailed Demand Data**:
       - View a table of hourly demand.
       - Changes are color-coded: green for increases, red for drops.
    5. In **Generation Breakdown**:
       - Select a day to see the composition of electricity sources.
       - Data is shown as a pie chart of daily average generation.

    **Notes:**
    - Data is preloaded from sample CSV files located in `sample_data/demand/` and `sample_data/generation/`.

    """)

# --- Get Base Path for Sample Data ---
def get_base_path(data_type="demand"):
    script_dir = os.path.dirname(__file__)
    if data_type == "demand":
        return os.path.join(script_dir, "sample_data", "demand")
    else:
        return os.path.join(script_dir, "sample_data", "generation")

# --- Load Demand Data ---
@st.cache_data(ttl=3600)
def load_demand_data():
    base_path = get_base_path("demand")
    all_files = glob.glob(os.path.join(base_path, "Custom-Report-2025-*-Seguimiento de la demanda de energía eléctrica (MW).csv"))
    if not all_files:
        st.error(f"No demand data files found in {base_path}.")
        return pd.DataFrame()

    full_df = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path, encoding="ISO-8859-1", sep=";", on_bad_lines='skip')
            raw_lines = df.iloc[:, 0].tolist()
            clean_lines = [line.strip() for line in raw_lines if line and "," in line]
            parsed_data = "\n".join(clean_lines)
            df_cleaned = pd.read_csv(StringIO(parsed_data), sep=",", quotechar='"')
            df_cleaned.reset_index(inplace=True)
            df_cleaned.rename(columns={"index": "Datetime"}, inplace=True)
            df_cleaned["Datetime"] = pd.to_datetime(df_cleaned["Datetime"], errors='coerce')
            df_cleaned.dropna(subset=["Datetime"], inplace=True)
            df_cleaned.set_index("Datetime", inplace=True)
            df_cleaned = df_cleaned[["Real", "Prevista"]]
            full_df.append(df_cleaned)
        except Exception as e:
            st.warning(f"Could not process {file_path}: {e}")

    if not full_df:
        st.error("All demand files failed to process.")
        return pd.DataFrame()

    combined = pd.concat(full_df).sort_index()
    combined["Daily Change"] = combined["Real"].pct_change()
    combined["Rolling Avg (30d)"] = combined["Real"].rolling(window=30, min_periods=1).mean()
    return combined

# --- Load Generation Data ---
@st.cache_data(ttl=3600)
def load_generation_data():
    base_path = get_base_path("generation")
    all_files = glob.glob(os.path.join(base_path, "Custom-Report-2025-*-Estructura de generación (MW)*.csv"))
    if not all_files:
        st.error(f"No generation files found in {base_path}.")
        return pd.DataFrame()

    full_gen_df = []
    failed_files = []

    for file_path in all_files:
        try:
            filename = os.path.basename(file_path)
            match = re.search(r"2025-(\d{1,2})-(\d{1,2})", filename)
            if not match:
                st.warning(f"Could not extract date from filename: {filename}")
                continue
            month, day = match.groups()
            report_date = f"2025-{int(month):02d}-{int(day):02d}"

            df = pd.read_csv(file_path, encoding="ISO-8859-1", sep=";", on_bad_lines='skip')
            raw_lines = df.iloc[:, 0].tolist()
            clean_lines = [line.strip() for line in raw_lines if "," in line]
            parsed_data = "\n".join(clean_lines)
            gen_df = pd.read_csv(StringIO(parsed_data))

            if "Hora" not in gen_df.columns:
                st.warning(f"Missing 'Hora' column in {filename}")
                continue

            if not gen_df["Hora"].astype(str).str.contains(":").any():
                gen_df["Hora"] = gen_df["Hora"].astype(str).str.zfill(4)
                gen_df["Hora"] = gen_df["Hora"].str.slice(0, 2) + ":" + gen_df["Hora"].str.slice(2, 4)

            gen_df["Hora"] = pd.to_datetime(report_date + " " + gen_df["Hora"].astype(str), errors="coerce")

            for col in gen_df.columns[1:]:
                if gen_df[col].dtype == object:
                    gen_df[col] = gen_df[col].str.replace('"', '', regex=False)
                gen_df[col] = pd.to_numeric(gen_df[col], errors="coerce")

            full_gen_df.append(gen_df)
        except Exception as e:
            failed_files.append((filename, str(e)))

    if not full_gen_df:
        st.error("All generation files failed to process.")
        return pd.DataFrame()

    gen_all = pd.concat(full_gen_df).dropna(subset=["Hora"]).sort_values("Hora")
    return gen_all

# --- Dashboard Toggle ---
if "dashboard_active" not in st.session_state:
    st.session_state.dashboard_active = False

if not st.session_state.dashboard_active:
    if st.button("Click to explore the dashboard"):
        st.session_state.dashboard_active = True

# --- Dashboard Main ---
if st.session_state.dashboard_active:
    data = load_demand_data()
    if data.empty:
        st.stop()

    gen_df = load_generation_data()
    if gen_df.empty or "Hora" not in gen_df.columns:
        st.stop()

    st.markdown("---")
    st.header("Trend Visualization")
    st.subheader("Electricity Demand – Real vs Forecast")

    unique_dates = sorted(list(set(data.index.date)))
    selected_date = st.selectbox("Select a specific day:", unique_dates, index=len(unique_dates)-1)

    day_data = data[data.index.date == selected_date]
    if not day_data.empty:
        min_time = day_data.index.min().time()
        max_time = day_data.index.max().time()
        start_time, end_time = st.slider(
            "Select time range (within selected day):",
            min_value=min_time,
            max_value=max_time,
            value=(min_time, max_time),
            format="HH:mm"
        )
        filtered_day_data = day_data.between_time(start_time.strftime("%H:%M"), end_time.strftime("%H:%M"))

        if not filtered_day_data.empty:
            fig = px.line(
                filtered_day_data.reset_index(), x="Datetime", y=["Real", "Prevista"],
                labels={"value": "MW", "Datetime": "Time", "variable": "Type"},
                title=f"Electricity Demand on {selected_date} from {start_time} to {end_time}",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("Key Performance Metrics")

    if not day_data.empty:
        latest_demand = float(day_data['Real'].iloc[-1])
        daily_change = day_data['Daily Change'].mean()
        volatility = day_data['Daily Change'].std()
        max_drop = float((day_data['Real'].max() - day_data['Real'].min()) / day_data['Real'].max())
        var_95 = day_data['Daily Change'].quantile(0.05)
    else:
        latest_demand = volatility = max_drop = var_95 = daily_change = float("nan")

    all_metrics = ["Latest Demand", "Volatility", "Maximum Drop", "VaR (95%)"]
    selected_metrics = st.multiselect("Select which metrics to display:", all_metrics, default=all_metrics)
    cols = st.columns(len(selected_metrics))
    for i, metric in enumerate(selected_metrics):
        if metric == "Latest Demand":
            cols[i].metric("Latest Demand", f"{latest_demand:,.0f} MW")
        elif metric == "Volatility":
            cols[i].metric("Volatility", f"{volatility*100:.2f}%" if not np.isnan(volatility) else "N/A")
        elif metric == "Maximum Drop":
            cols[i].metric("Maximum Drop", f"{max_drop*100:.2f}%" if not np.isnan(max_drop) else "N/A")
        elif metric == "VaR (95%)":
            cols[i].metric("VaR (95%)", f"{var_95*100:.2f}%" if not np.isnan(var_95) else "N/A")

    st.markdown("---")
    st.header("Detailed Demand Data")
    with st.expander("Show Historical Data Table with Conditional Formatting"):
        styled_df = data.copy().dropna()
        styled_df['Daily Change (%)'] = styled_df['Daily Change'] * 100
        styled_df['Daily Change (%)'] = styled_df['Daily Change (%)'].map(lambda x: f"{x:.2f}%")
        styled_df = styled_df.reset_index()[['Datetime', 'Real', 'Prevista', 'Daily Change (%)']]

        def highlight_change(val):
            try:
                return f"color: {'red' if float(val.replace('%','')) < 0 else 'green'}"
            except:
                return ""

        st.dataframe(
            styled_df.style.applymap(highlight_change, subset=['Daily Change (%)']),
            use_container_width=True
        )

    st.markdown("---")
    st.header("Generation Breakdown")
    st.subheader("Generation Mix by Day")

    gen_df["Date"] = gen_df["Hora"].dt.date
    available_dates = sorted(gen_df["Date"].unique())
    selected_date_gen = st.selectbox("Select Date for Generation Snapshot:", available_dates, index=len(available_dates)-1)

    daily_data = gen_df[gen_df["Date"] == selected_date_gen]
    mix_cols = [
        "Eólica", "Nuclear", "Carbón", "Ciclo combinado",
        "Solar fotovoltaica", "Solar térmica", "Térmica renovable",
        "Motores diésel", "Turbina de gas", "Turbina de vapor",
        "Generación auxiliar", "Cogeneración y residuos"
    ]
    daily_avg = daily_data[mix_cols].mean().to_frame(name="MW")
    daily_avg = daily_avg[daily_avg["MW"] > 0]

    fig_pie = px.pie(daily_avg, values="MW", names=daily_avg.index,
                     title=f"Electricity Generation Sources – Daily Average ({selected_date_gen})")
    st.plotly_chart(fig_pie, use_container_width=True)
