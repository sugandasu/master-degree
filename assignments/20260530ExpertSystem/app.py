import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

# =====================================================================
# 1. 3-INPUT, 3-MF ANFIS ARCHITECTURE (27 Rules)
# =====================================================================
class ANFIS(nn.Module):
    def __init__(self, num_inputs=3, num_mf=3): # Configured for 3 MFs
        super(ANFIS, self).__init__()
        self.num_inputs = num_inputs
        self.num_mf = num_mf
        self.num_rules = num_mf ** num_inputs
        
        self.mf_means = nn.Parameter(torch.randn(num_inputs, num_mf))
        self.mf_sigmas = nn.Parameter(torch.ones(num_inputs, num_mf))
        self.consequent_params = nn.Parameter(torch.randn(self.num_rules, num_inputs + 1))
        
        grids = torch.meshgrid([torch.arange(num_mf) for _ in range(num_inputs)], indexing='ij')
        self.rule_grid = torch.stack([g.flatten() for g in grids], dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(2).repeat(1, 1, self.num_mf)
        means = self.mf_means.unsqueeze(0).repeat(batch_size, 1, 1)
        sigmas = self.mf_sigmas.unsqueeze(0).repeat(batch_size, 1, 1)
        
        w_inputs = torch.exp(-0.5 * ((x_expanded - means) / (sigmas + 1e-8)) ** 2)
        
        w_rules = torch.zeros(batch_size, self.num_rules, device=x.device)
        for i in range(self.num_rules):
            rule_idx = self.rule_grid[i]
            w_rules[:, i] = torch.prod(torch.stack([w_inputs[:, j, rule_idx[j]] for j in range(self.num_inputs)], dim=1), dim=1)
        
        w_sum = torch.sum(w_rules, dim=1, keepdim=True) + 1e-8
        w_normalized = w_rules / w_sum
        
        x_biased = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)
        rule_outputs = torch.matmul(x_biased, self.consequent_params.T)
        
        return torch.sum(w_normalized * rule_outputs, dim=1, keepdim=True)

# =====================================================================
# 2. CACHE & LOAD PIPELINE WITH DYNAMIC DIRECTORY RESOLUTION
# =====================================================================
@st.cache_resource
def load_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "anfis_concrete_model.pth")
    scalers_path = os.path.join(current_dir, "concrete_scalers.pkl")
    
    # Initialize using 3 inputs and 3 MFs
    model = ANFIS(num_inputs=3, num_mf=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    pipeline = joblib.load(scalers_path)
    return model, pipeline['scaler_X'], pipeline['scaler_y']

try:
    anfis_model, scaler_X, scaler_y = load_pipeline()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_trace = e

# =====================================================================
# 3. INTERACTIVE DASHBOARD INTERFACE
# =====================================================================
st.set_page_config(page_title="Concrete Predictor (3 MF)", layout="centered")
st.title("🏗️ Concrete Compressive Strength Predictor")
st.write("ANFIS Model running 3 Features with 3 Membership Functions each (27 Fuzzy Rules).")

if not model_loaded:
    st.error("🚨 System initialization failure! Check file matching or retrain the model.")
    st.exception(error_trace)
else:
    st.markdown("---")
    st.subheader("🔧 Mix Properties")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        cement = st.slider("Cement ($kg/m^3$)", min_value=100.0, max_value=550.0, value=300.0, step=5.0)
    with col2:
        water = st.slider("Water ($kg/m^3$)", min_value=120.0, max_value=250.0, value=180.0, step=2.0)
    with col3:
        age = st.slider("Age (Curing Days)", min_value=1, max_value=365, value=28, step=1)

    raw_inputs = np.array([[cement, water, age]])

    if st.button("🔮 Calculate Compressive Strength", type="primary"):
        inputs_scaled = scaler_X.transform(raw_inputs)
        inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction_scaled = anfis_model(inputs_tensor).numpy()

        final_strength = scaler_y.inverse_transform(prediction_scaled)[0][0]

        st.markdown("---")
        st.subheader("📊 Estimated Output")
        st.metric(label="Target Compressive Strength", value=f"{final_strength:.2f} MPa")
        
        if final_strength >= 35:
            st.success("High Performance Structural Blend")
        elif final_strength >= 20:
            st.info("Standard Residential Utility Grade")
        else:
            st.warning("Low Load Concrete / Non-Structural Lean Mix")