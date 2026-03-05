import streamlit as st
import json
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
from fpdf import FPDF
import io
from datetime import datetime
import base64

st.set_page_config(page_title="Marine Power Management", page_icon="🚢", layout="wide")

# Custom CSS for UI
st.markdown("""
<style>
.metric-box {
    background-color: #1E1E1E;
    color: #FFFFFF;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    with open('dg_data.json', 'r') as f:
        return json.load(f)

try:
    dg_data = load_data()
except FileNotFoundError:
    st.error("dg_data.json not found! Please run generate_data.py first.")
    st.stop()

st.title("🚢 Marine Generator Performance & Fuel Analytics Suite")
st.markdown("Advanced analytics utilizing Cubic Spline Interpolation, ISO 3046-1 Atmospheric Corrections, and dynamic risk monitoring.")

# --- SIDEBAR inputs ---
st.sidebar.header("Ship Details")
ship_name = st.sidebar.text_input("Vessel Name", "MV Explorer")

st.sidebar.header("1. Engine Specifications")
makes = ["Manual/Custom"] + sorted(list(dg_data.keys()))
make = st.sidebar.selectbox("Manufacturer", makes)

spec_url = ""
if make == "Manual/Custom":
    model = "Custom Engine"
    rated_kw = st.sidebar.number_input("Rated Power (kW)", min_value=10.0, value=100.0, step=10.0)
    rpm = "N/A"
    cooling = "N/A"
else:
    models = sorted(list(dg_data[make].keys()))
    model = st.sidebar.selectbox("Model", models)
    engine_details = dg_data[make][model]
    rated_kw = engine_details["Rated_kW"]
    rpm = engine_details["RPM"]
    cooling = engine_details["Cooling_Type"]
    spec_url = engine_details.get("Spec_Sheet_URL", "")
    
    st.sidebar.markdown(f"**Rated Power:** {rated_kw} kW")
    st.sidebar.markdown(f"**RPM:** {rpm}")
    st.sidebar.markdown(f"**Cooling:** {cooling}")
    if spec_url:
        st.sidebar.markdown(f"[📄 View Technical Data Sheet]({spec_url})")

if "peak_load" not in st.session_state:
    st.session_state["peak_load"] = 25.0

if st.session_state["peak_load"] > rated_kw:
    st.sidebar.error("Error: Load cannot exceed Generator Capacity")
    peak_load = st.sidebar.number_input("Peak Load (kW)", min_value=0.0, step=10.0, key="peak_load")
    st.stop()
else:
    peak_load = st.sidebar.number_input("Peak Load (kW)", min_value=0.0, max_value=float(rated_kw), step=10.0, key="peak_load")


st.sidebar.header("2. ISO 3046-1 Corrections")
ambient_temp = st.sidebar.slider("Ambient Temp (°C)", -10, 55, 25)
baro_pressure = st.sidebar.slider("Barometric Pressure (mbar)", 800, 1100, 1000)

st.sidebar.header("3. Fuel Chemistry")
fuel_grades = {
    "MGO (Marine Gas Oil)": {"density": 0.85, "lhv": 42700, "cef": 3.206},
    "MDO (Marine Diesel Oil)": {"density": 0.89, "lhv": 40000, "cef": 3.206},
    "HFO (Heavy Fuel Oil)": {"density": 0.96, "lhv": 39000, "cef": 3.114}
}
fuel_choice = st.sidebar.selectbox("Fuel Type", list(fuel_grades.keys()))
fuel_props = fuel_grades[fuel_choice]
fuel_density = fuel_props["density"]
actual_lhv = fuel_props["lhv"]
fuel_cef = fuel_props["cef"]  # kg CO2 per kg of fuel
ref_lhv = 42700 # Reference LHV in kJ/kg

# --- CALCULATIONS ---
load_pct = (peak_load / rated_kw) * 100 if rated_kw > 0 else 0

# ISO 3046 basic penalty factor. Ref: 25C, 1000mbar.
temp_penalty = max(0.0, (ambient_temp - 25) * 0.002) # +0.2% penalty per deg over 25C
pressure_penalty = max(0.0, (1000 - baro_pressure) * 0.0001) # +0.01% penalty per mbar under 1000
env_factor = 1.0 + temp_penalty + pressure_penalty

# Fuel Heating Value Correction Factor
lhv_factor = ref_lhv / actual_lhv

# L/hr series generation for charts
x_pct = np.linspace(0, 100, 50)
x_kw = x_pct / 100 * rated_kw
y_l_hr_series = []
y_sfoc_series = [] # To display specific consumption curve

if make == "Manual/Custom":
    # Willans Line
    base_kg_hr = (0.02 * rated_kw) + (0.18 * peak_load)
    corrected_kg_hr = base_kg_hr * env_factor * lhv_factor
    consumption_l_hr = corrected_kg_hr / fuel_density
    
    for p in x_pct:
        l_kw = (p / 100.0) * rated_kw
        b_kg = (0.02 * rated_kw) + (0.18 * l_kw)
        c_kg = b_kg * env_factor * lhv_factor
        y_l_hr_series.append(c_kg / fuel_density)
        y_sfoc = (c_kg * 1000 / l_kw) if l_kw > 0 else 0
        y_sfoc_series.append(y_sfoc)
        
    method_used = "Willans Line Method (Generic Engine)"
else:
    # Cubic Spline Interpolation
    curves = engine_details["SFOC_Curve"]
    x_val = np.array([10, 25, 50, 75, 100])
    
    sfoc_100 = curves["100"]
    sfoc_10 = sfoc_100 * 1.5 # extrapolate idle inefficiency
    raw_sfoc = np.array([sfoc_10, curves["25"], curves["50"], curves["75"], curves["100"]])
    
    cons_kg = (x_val / 100 * rated_kw) * raw_sfoc / 1000
    
    spline = CubicSpline(x_val, cons_kg, bc_type='natural')
    
    base_kg_hr = float(spline(max(10.0, load_pct))) # clamp low end
    if load_pct < 10:
        base_kg_hr = base_kg_hr * (load_pct / 10)
        
    corrected_kg_hr = base_kg_hr * env_factor * lhv_factor
    consumption_l_hr = corrected_kg_hr / fuel_density
    
    for p in x_pct:
        if p < 10:
            c_kg = float(spline(10.0)) * (p / 10.0)
        else:
            c_kg = float(spline(p))
            
        c_kg_corr = c_kg * env_factor * lhv_factor
        l_kw = (p / 100.0) * rated_kw
        
        y_l_hr_series.append(c_kg_corr / fuel_density)
        y_sfoc = (c_kg_corr * 1000 / l_kw) if l_kw > 0 else 0
        y_sfoc_series.append(y_sfoc)

    method_used = "Non-Linear Cubic Spline Interpolation"

mass_flow_kg_hr = consumption_l_hr * fuel_density

# Carbon Intensity (gCO2/kWh)
if peak_load > 0:
    carbon_intensity_g_kwh = (mass_flow_kg_hr * fuel_cef * 1000) / peak_load
else:
    carbon_intensity_g_kwh = 0

# --- MAIN DASHBOARD ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    fig_load = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = load_pct,
        title = {'text': "Load Factor (%)", 'font': {'color': 'white'}},
        number = {'font': {'color': 'white'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': 'white'},
            'bar': {'color': "cyan"},
            'steps' : [
                {'range': [0, 30], 'color': "rgba(255, 0, 0, 0.4)"},
                {'range': [30, 95], 'color': "rgba(0, 255, 0, 0.3)"},
                {'range': [95, 100], 'color': "rgba(255, 165, 0, 0.6)"}
            ],
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 95}
        }
    ))
    fig_load.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=250, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_load, use_container_width=True)

with col2:
    fig_carbon = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = carbon_intensity_g_kwh,
        title = {'text': "Carbon Intensity (gCO2/kWh)", 'font': {'color': 'white'}},
        number = {'font': {'color': 'white'}, 'valueformat': '.0f'},
        gauge = {
            'axis': {'range': [None, 1500], 'tickcolor': 'white'},
            'bar': {'color': "orange"},
            'steps' : [
                {'range': [0, 600], 'color': "rgba(0, 255, 0, 0.2)"},
                {'range': [600, 900], 'color': "rgba(255, 255, 0, 0.2)"},
                {'range': [900, 1500], 'color': "rgba(255, 0, 0, 0.3)"}
            ],
        }
    ))
    fig_carbon.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=250, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_carbon, use_container_width=True)

with col3:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box'><h3>ISO Corrected Fuel Flow</h3><h2>{consumption_l_hr:.1f} L/hr</h2></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box' style='margin-top:10px;'><h3>Calculation Engine</h3><p>{method_used}</p></div>", unsafe_allow_html=True)

st.divider()

# --- OPERATIONAL RISK MODULE ---
st.subheader("🛡️ Safety Diagnostics & Operational Risk")
risk_cols = st.columns(2)
with risk_cols[0]:
    if load_pct < 30:
        st.error("**🛑 WET STACKING RISK:** Engine load is under 30%. Operating at this load promotes unburnt fuel glazing and severe carbon buildup.")
    elif load_pct > 95:
        st.error("**⚠️ OVERLOAD ALERT:** Engine load is above 95%. Sustained operation may violate Continuous Service Rating (CSR) causing critical thermal stresses.")
    else:
        st.success("**✅ OPTIMAL LOAD:** Engine operates within the recommended efficiency health band.")

with risk_cols[1]:
    if env_factor > 1.05:
        st.warning(f"**🌡️ ISO 3046 DERATING:** Ambient/Atmospheric penalty is increasing fuel consumption by {((env_factor - 1) * 100):.1f}%.")
    else:
        st.info("**🌿 ATMOSPHERIC:** Operating near standard ISO reference conditions.")

# --- CHART & LOGS ---
col_chart, col_logs = st.columns([2, 1])

with col_chart:
    st.subheader("Specific Fuel Consumption Curve")
    # Hide 0% load to avoid infinity spikes in SFOC curve display
    valid_plot_idx = x_pct > 5
    filtered_x = x_pct[valid_plot_idx]
    filtered_y = np.array(y_sfoc_series)[valid_plot_idx]
    
    current_sfoc = (mass_flow_kg_hr * 1000 / peak_load) if peak_load > 0 else 0
    
    chart_data = pd.DataFrame({
        "Load (%)": filtered_x,
        "SFOC (g/kWh)": filtered_y
    })
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=chart_data["Load (%)"], y=chart_data["SFOC (g/kWh)"], mode='lines', name='SFOC Model', line=dict(color='lightgreen', width=3)))
    if peak_load > (0.05 * rated_kw):
        fig_line.add_trace(go.Scatter(x=[load_pct], y=[current_sfoc], mode='markers', name='Current OPEX', marker=dict(color='red', size=12)))
    
    fig_line.update_layout(
        xaxis_title="Engine Load (%)",
        yaxis_title="Specific Fuel Consumption (g/kWh)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    fig_line.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig_line.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig_line, use_container_width=True)

with col_logs:
    st.subheader("Data Export & Verification")
    observations = st.text_area("Engineer's Remarks", "Inspected exhaust manifold. Temperatures nominal. Ready for PDF export.")
    
    # --- PDF EXPORT ---
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=15, style="B")
        pdf.cell(200, 10, txt="MARINE GENERATOR PERFORMANCE & FUEL ANALYTICS", ln=1, align='C')
        pdf.set_font("Helvetica", size=10)
        pdf.cell(200, 10, txt=f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=1, align='C')
        
        pdf.ln(10)
        pdf.set_font("Helvetica", size=12, style="B")
        pdf.cell(200, 10, txt="Ship & Engine Specifications", ln=1)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(200, 8, txt=f"Vessel Name: {ship_name}", ln=1)
        pdf.cell(200, 8, txt=f"Manufacturer: {make}", ln=1)
        if make != "Manual/Custom":
            pdf.cell(200, 8, txt=f"Model: {model}", ln=1)
        pdf.cell(200, 8, txt=f"Rated Power: {rated_kw} kW", ln=1)
        pdf.cell(200, 8, txt=f"Fuel Type: {fuel_choice} (LHV: {actual_lhv} kJ/kg)", ln=1)
        
        pdf.ln(5)
        pdf.set_font("Helvetica", size=12, style="B")
        pdf.cell(200, 10, txt="ISO Corrected Operating Parameters", ln=1)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(200, 8, txt=f"Peak Load: {peak_load} kW ({load_pct:.1f}% Load Factor)", ln=1)
        pdf.cell(200, 8, txt=f"Ambient Temp: {ambient_temp} C | Barometric Press: {baro_pressure} mbar", ln=1)
        pdf.cell(200, 8, txt=f"Engine Strategy: {method_used}", ln=1)
        
        pdf.ln(5)
        pdf.set_font("Helvetica", size=12, style="B")
        pdf.cell(200, 10, txt="Analytics Results", ln=1)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(200, 8, txt=f"Calculated ISO Fuel Flow: {consumption_l_hr:.2f} L/hr", ln=1)
        pdf.cell(200, 8, txt=f"Mass Flow Rate: {mass_flow_kg_hr:.2f} kg/hr", ln=1)
        pdf.cell(200, 8, txt=f"Carbon Intensity: {carbon_intensity_g_kwh:.0f} gCO2/kWh", ln=1)
        
        pdf.ln(5)
        pdf.set_font("Helvetica", size=12, style="B")
        pdf.cell(200, 10, txt="Engineer's Remarks", ln=1)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 8, txt=observations)
        
        return bytes(pdf.output())

    pdf_bytes = generate_pdf()
    
    st.download_button(
        label="📄 Download Advanced PDF Report",
        data=pdf_bytes,
        file_name=f"{ship_name.replace(' ', '_')}_Performance_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
