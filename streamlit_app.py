import requests
import streamlit as st

st.title("ImmoPrix — Test de l’API (California Housing)")

API_URL = st.text_input("URL de l'API", "http://127.0.0.1:8000")

st.subheader("Features")
col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input("MedInc", value=3.0)
    HouseAge = st.number_input("HouseAge", value=20.0)
    AveRooms = st.number_input("AveRooms", value=5.0)
    AveBedrms = st.number_input("AveBedrms", value=1.0)

with col2:
    Population = st.number_input("Population", value=1000.0)
    AveOccup = st.number_input("AveOccup", value=3.0)
    Latitude = st.number_input("Latitude", value=34.0)
    Longitude = st.number_input("Longitude", value=-118.0)

payload = {
    "MedInc": MedInc,
    "HouseAge": HouseAge,
    "AveRooms": AveRooms,
    "AveBedrms": AveBedrms,
    "Population": Population,
    "AveOccup": AveOccup,
    "Latitude": Latitude,
    "Longitude": Longitude,
}

# --------------------
# Health check
# --------------------
if st.button("Tester /health"):
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        r.raise_for_status()
        st.success("✅ API OK")
        st.json(r.json())
    except Exception as e:
        st.error(f"Erreur /health: {e}")

# --------------------
# Prediction
# --------------------
if st.button("Prédire"):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        st.success(
            f" Prediction MedHouseVal = {data['prediction']:.4f} (x100k $)"
        )
        st.caption(f"Model URI: {data['model_uri']}")
    except Exception as e:
        st.error(f"Erreur /predict: {e}")
