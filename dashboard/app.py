import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from pathlib import Path
import numpy as np



COLUMN_NAMES = ['engine_id', 'cycle',
                'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

#  LOAD MODEL & DATA 

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent / 'model' / 'rf_model.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_test_data():
    data_path = Path(__file__).parent.parent / 'data' / 'test_FD001.txt'
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    df = df.dropna(axis=1, how='all')
    return df

#  PREDICTION 

def predict_health_ratio(model, engine_data, features):
    """
    Predict normalized RUL (health ratio in range 0–1)
    """
    engine_df = engine_data.copy()
    engine_df = engine_df.sort_values('cycle').reset_index(drop=True)

    X = engine_df[features]

    # Model predicts RUL_RATIO (0–1)
    rul_ratio = model.predict(X)

    # Safety clamp
    rul_ratio = np.clip(rul_ratio, 0, 1)

    engine_df['health_pct'] = rul_ratio * 100
    engine_df['health_pct'] = engine_df['health_pct'].clip(0, 100)

    return engine_df[['cycle', 'health_pct']]

def get_health_status(health):
    if health > 80:
        return "Healthy", "green"
    elif health >= 40:
        return "Warning", "orange"
    else:
        return "Critical", "red"

# MAIN 

def main():
    st.set_page_config(
        page_title="Jet Engine Predictive Maintenance",
        page_icon="✈️",
        layout="wide"
    )

    st.title("✈️ Jet Engine Predictive Maintenance Dashboard")
    st.markdown("**NASA CMAPSS FD001 — Normalized RUL Model**")
    st.markdown("---")

    model_data = load_model()
    model = model_data['model']
    features = model_data['features']

    test_data = load_test_data()

    engine_ids = sorted(test_data['engine_id'].unique())

    # SIDEBAR 

    st.sidebar.header("⚙️ Engine Selection")
    selected_engine = st.sidebar.selectbox("Select Engine ID", engine_ids)
    lite_mode = st.sidebar.checkbox("Lite mode (faster)", value=True, help="Render fewer points and smaller tables for low-end machines")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Health is predicted using **normalized RUL**.\n"
        "0% = end of lifecycle\n"
        "100% = start of lifecycle"
    )

    #  ENGINE DATA 

    engine_data = test_data[test_data['engine_id'] == selected_engine].copy()
    engine_data = engine_data.sort_values('cycle').reset_index(drop=True)

    predictions = predict_health_ratio(model, engine_data, features)

    current = predictions.iloc[-1]
    current_cycle = int(current['cycle'])
    current_health = float(current['health_pct'])

    status_text, status_color = get_health_status(current_health)

    #  METRICS 

    st.subheader("📊 Current Engine Status")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Engine ID", f"#{selected_engine}")
    c2.metric("Current Cycle", current_cycle)
    c3.metric("Health %", f"{current_health:.1f}%")
    c4.metric("Status", status_text)

    # Show model validation RMSE if available in model metadata
    rmse_cycles = model_data.get('rmse_validation_cycles') if isinstance(model_data, dict) else None
    if rmse_cycles is not None:
        st.metric("Model RMSE (validation, cycles)", f"{rmse_cycles:.2f}")

    # Status messages
    if status_text == "Healthy":
        st.success("✅ Engine operating normally")
    elif status_text == "Warning":
        st.warning("⚠️ Engine degradation detected — monitor closely")
    else:
        st.error("🚨 Critical condition — maintenance required")



    # Maintenance outlook (very rough estimate)
    st.markdown("### ⏳ Maintenance Outlook")

# Use current cycle instead of engine_data
    estimated_remaining_cycles = int(
        (current_health / 100) * current_cycle
    )

    if status_text == "Critical":
        st.error(
            f"🚨 **Maintenance strongly recommended within ~{estimated_remaining_cycles} cycles**"
        )
    elif status_text == "Warning":
        st.warning(
            f"⚠️ **Plan maintenance within ~{estimated_remaining_cycles} cycles**"
        )
    else:
        st.success(
            f"✅ **No immediate maintenance required**"
        )


    #  CHARTS 
    st.markdown("---")
    st.subheader("📈 Health Over Time")
    # Downsample plot points in lite mode for faster rendering
    plot_df = predictions
    if lite_mode and len(predictions) > 50:
        idx = np.linspace(0, len(predictions) - 1, 50, dtype=int)
        plot_df = predictions.iloc[idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df['cycle'],
        y=plot_df['health_pct'],
        mode='lines+markers' if not lite_mode else 'lines',
        name='Health %',
        line=dict(width=3)
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Healthy Threshold")
    fig.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Critical Threshold")
    fig.update_layout(xaxis_title="Cycle", yaxis_title="Health (%)", yaxis=dict(range=[0, 100]), height=450)
    st.plotly_chart(fig, use_container_width=True)

    #  DATA TABLE
    st.markdown("---")
    # Detailed predictions (hidden by default in lite mode)
    show_details = st.checkbox("Show detailed predictions", value=not lite_mode)
    if show_details:
        df_show = predictions.copy()
        df_show['cycle'] = df_show['cycle'].astype(int)
        df_show['health_pct'] = df_show['health_pct'].round(2)
        df_show.columns = ['Cycle', 'Health %']
        max_rows = 100 if lite_mode else None
        st.dataframe(df_show.head(max_rows), use_container_width=True)

    st.markdown("---")
    st.caption(
        "🛠️ Predictive Maintenance Dashboard | RandomForest trained on Normalized RUL | Hackathon-ready implementation"
    )

    # Fleet context (relative view)
    st.markdown("### 🏭 Fleet Context (Relative View)")
    fleet_last = (
        test_data
        .sort_values('cycle')
        .groupby('engine_id')
        .tail(1)
        .copy()
    )
    fleet_pred = predict_health_ratio(model, fleet_last, features)
    fleet_pred['engine_id'] = fleet_last['engine_id'].values
    fleet_pred = fleet_pred.sort_values('health_pct')
    fleet_display_count = 5 if lite_mode else 10
    st.dataframe(
        fleet_pred.rename(columns={'cycle': 'Last Cycle', 'health_pct': 'Health %'}).head(fleet_display_count),
        use_container_width=True
    )


if __name__ == "__main__":
    main()
