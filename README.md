# UK Water–Energy Nexus: Resilience Gap Dashboard

This Streamlit app quantifies **resilience gaps** across UK regions at the water–energy nexus (Objective 1).  
It provides scenario knobs, user-defined weights, KMeans clustering, and rankings. Bring your own CSV to replace the sample data.

## Quick Start

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run app
streamlit run streamlit_app.py
```

## Data

A sample dataset is included at `data/resilience_sample_uk.csv`.  
**Expected columns** if you upload your own CSV:

```
region,
flood_risk_index,
drought_risk_index,
heatwave_days,
water_supply_deficit,
energy_demand_growth,
interdependence_index,
adaptive_capacity,
critical_infra_score,
population_m,
gva_billion,
adaptation_capex_gap
```

All numeric fields are treated as continuous variables. The **resilience_gap_score** is computed as:

```
gap = w_risk * mean(normalized {flood_risk_index, drought_risk_index, heatwave_days})
    + w_exposure * mean(normalized {water_supply_deficit, energy_demand_growth, critical_infra_score, interdependence_index})
    - w_capacity * normalized(adaptive_capacity)
```

Change weights and scenarios from the sidebar. Scenario adjustments are illustrative — replace with UKCP18/CMIP6/NGFS-based deltas for rigor.

## Notes & Extensibility

- Replace synthetic data with Ofwat/Ofgem/Met Office datasets.
- Add geographic mapping via GeoPandas/TopoJSON when shapes are available.
- Calibrate weights through stakeholder workshops or analytic hierarchy process (AHP).
- Track uncertainty bands via bootstrapping or Bayesian models.

## License
For demonstration and academic use.
