
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(
    page_title="UK Water–Energy Nexus: Resilience Gap Dashboard",
    page_icon="💧⚡",
    layout="wide"
)

@st.cache_data
def load_data(default_path: str):
    df = pd.read_csv(default_path)
    return df

def normalize_columns(df, cols):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x = scaler.fit_transform(df[cols])
    return x, scaler

def compute_resilience_gap(df, weights):
    risk_cols = ["flood_risk_index", "drought_risk_index", "heatwave_days"]
    exposure_cols = ["water_supply_deficit", "energy_demand_growth", "critical_infra_score", "interdependence_index"]
    capacity_col = ["adaptive_capacity"]
    all_cols = risk_cols + exposure_cols + capacity_col
    x_norm, _ = normalize_columns(df, all_cols)
    X = pd.DataFrame(x_norm, columns=all_cols)
    gap = (
        weights["w_risk"] * X[risk_cols].mean(axis=1) +
        weights["w_exposure"] * X[exposure_cols].mean(axis=1) -
        weights["w_capacity"] * X[capacity_col].iloc[:,0]
    )
    return gap

def scenario_shift(df, scenario):
    df = df.copy()
    if scenario == "2030 (High Warming)":
        df["heatwave_days"] *= 1.25
        df["drought_risk_index"] *= 1.15
        df["flood_risk_index"] *= 1.10
        df["energy_demand_growth"] *= 1.20
        df["water_supply_deficit"] *= 1.30
        df["adaptive_capacity"] *= 0.95
    elif scenario == "2050 (High Warming)":
        df["heatwave_days"] *= 1.6
        df["drought_risk_index"] *= 1.35
        df["flood_risk_index"] *= 1.25
        df["energy_demand_growth"] *= 1.35
        df["water_supply_deficit"] *= 1.55
        df["adaptive_capacity"] *= 0.9
    elif scenario == "2030 (Managed Adaptation)":
        df["heatwave_days"] *= 1.1
        df["drought_risk_index"] *= 1.05
        df["flood_risk_index"] *= 1.05
        df["energy_demand_growth"] *= 1.10
        df["water_supply_deficit"] *= 1.15
        df["adaptive_capacity"] *= 1.05
    return df

st.sidebar.header("Scenario & Weights")
scenario = st.sidebar.selectbox(
    "Select scenario",
    ["Baseline (Today)", "2030 (Managed Adaptation)", "2030 (High Warming)", "2050 (High Warming)"],
    index=0
)

st.sidebar.subheader("Weighting (0–2)")
w_risk = st.sidebar.slider("Risk weight", 0.0, 2.0, 1.0, 0.1)
w_exposure = st.sidebar.slider("Exposure weight", 0.0, 2.0, 1.0, 0.1)
w_capacity = st.sidebar.slider("Adaptive capacity weight (reduces gap)", 0.0, 2.0, 1.0, 0.1)
weights = {"w_risk": w_risk, "w_exposure": w_exposure, "w_capacity": w_capacity}

st.sidebar.subheader("Clustering")
n_clusters = st.sidebar.slider("Number of clusters (KMeans)", 2, 6, 3, 1)

st.sidebar.subheader("Data")
uploaded = st.sidebar.file_uploader("Upload custom CSV (optional)", type=["csv"])
st.sidebar.markdown("""
**Expected columns:**  
`region, flood_risk_index, drought_risk_index, heatwave_days, water_supply_deficit, energy_demand_growth, interdependence_index, adaptive_capacity, critical_infra_score, population_m, gva_billion, adaptation_capex_gap`
""")

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = load_data("data/resilience_sample_uk.csv")

df_scn = df.copy()
if scenario != "Baseline (Today)":
    df_scn = scenario_shift(df_scn, scenario)

gap = compute_resilience_gap(df_scn, weights)
df_scn["resilience_gap_score"] = gap
df_scn["gap_rank"] = df_scn["resilience_gap_score"].rank(ascending=False, method="min").astype(int)

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(df_scn[["flood_risk_index","drought_risk_index","heatwave_days","water_supply_deficit","energy_demand_growth","critical_infra_score","interdependence_index","adaptive_capacity"]])
km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
clusters = km.fit_predict(X)
df_scn["cluster"] = clusters

pca = PCA(n_components=2, random_state=42)
xy = pca.fit_transform(X)
df_scn["pc1"] = xy[:,0]
df_scn["pc2"] = xy[:,1]

st.title("💧⚡ UK Water–Energy Nexus: Resilience Gap Dashboard")
st.caption("Objective 1: Quantify resilience gaps across UK regions — configurable scenarios, weights, clustering, and rankings.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Regions", len(df_scn))
with c2:
    st.metric("Highest Gap (score)", f"{df_scn['resilience_gap_score'].max():.2f}")
with c3:
    st.metric("Lowest Gap (score)", f"{df_scn['resilience_gap_score'].min():.2f}")
with c4:
    st.metric("Avg Adaptive Capacity", f"{df_scn['adaptive_capacity'].mean():.1f}")

st.subheader("Resilience Gap by Region (Higher = Worse)")
fig_bar = px.bar(
    df_scn.sort_values("resilience_gap_score", ascending=False),
    x="region", y="resilience_gap_score", color="cluster",
    hover_data=["gap_rank","adaptive_capacity","flood_risk_index","drought_risk_index","water_supply_deficit","energy_demand_growth","interdependence_index","critical_infra_score"],
    title="Resilience Gap Ranking"
)
fig_bar.update_layout(xaxis_title="", yaxis_title="Gap Score (unitless)")
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Risk–Exposure Map (Bubble size = adaptive capacity)")
fig_scatter = px.scatter(
    df_scn,
    x="drought_risk_index",
    y="flood_risk_index",
    size="adaptive_capacity",
    color="cluster",
    hover_name="region",
    hover_data=["resilience_gap_score","gap_rank","heatwave_days","water_supply_deficit","energy_demand_growth"],
    title="Risk Interaction Map"
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("PCA View of Regional Profiles")
fig_pca = px.scatter(
    df_scn, x="pc1", y="pc2", color="cluster", hover_name="region",
    hover_data=["resilience_gap_score","adaptive_capacity","water_supply_deficit","energy_demand_growth","interdependence_index"],
    title="PCA Projection (features standardized)"
)
st.plotly_chart(fig_pca, use_container_width=True)

st.subheader("Detailed Table")
show_cols = [
    "region","gap_rank","resilience_gap_score","flood_risk_index","drought_risk_index","heatwave_days",
    "water_supply_deficit","energy_demand_growth","interdependence_index","adaptive_capacity",
    "critical_infra_score","population_m","gva_billion","adaptation_capex_gap","cluster"
]
st.dataframe(df_scn[show_cols].sort_values("gap_rank"))

st.download_button(
    "Download results (CSV)",
    df_scn[show_cols].sort_values("gap_rank").to_csv(index=False),
    file_name="resilience_gap_results.csv",
    mime="text/csv"
)

st.info("""
**Method (illustrative):**  
Resilience gap score = `w_risk * mean(normalized {flood, drought, heatwave}) + w_exposure * mean(normalized {deficit, demand, infra, interdependence}) - w_capacity * normalized(adaptive_capacity)`  
Adjust scenario & weights in the sidebar. Upload your CSV to replace the sample data.
""")

st.caption("This is a demonstrator. Replace sample indicators with authoritative sources such as UKCP18/CMIP6 for climate hazards, Ofwat/Ofgem datasets for infrastructure and demand, and local resilience strategies for adaptive capacity.")
