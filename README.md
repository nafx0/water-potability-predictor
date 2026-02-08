---
title: Water Potability Predictor
emoji: 💧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 💧 Water Potability Predictor

Ensure the water you consume is safe — instantly.

---

## Project Overview

Access to safe drinking water is critical for health. Contaminated water can lead to serious illnesses. This project provides an **AI-powered tool** that predicts whether water is potable based on its chemical properties. It can be used by households, researchers, or water quality agencies to quickly assess water safety.

---

## Why It Matters

- **Health Safety:** Prevents waterborne diseases by identifying unsafe water.
- **Quick Assessment:** Provides instant predictions without lab tests.
- **Educational Insight:** Demonstrates how AI can assist in environmental monitoring.

---

## How It Works

1. **Data Input:** Users provide water sample measurements, such as pH, hardness, solids, chloramines, and turbidity.
2. **Feature Engineering:** The system derives important ratios (e.g., solids-to-hardness) to enhance prediction accuracy.
3. **Outlier Handling:** Extreme values are automatically adjusted to avoid skewed results.
4. **Prediction:** A trained AI model classifies the water as **Potable** or **Not Potable** with an associated probability.

---

## Technologies Used

- **Python 3.10**
- **Gradio:** Interactive web interface
- **Scikit-learn:** Machine learning model
- **Pandas & NumPy:** Data processing
- **Joblib:** Model serialization

---

## How to Use

1. Access the web interface.
2. Adjust the sliders to enter your water sample measurements.
3. Click **Predict** to get the result:  
   - **Potable** ✅  
   - **Not Potable** ❌

---

## Project Impact

This tool bridges the gap between **technical AI models** and **practical everyday use**. Users don’t need to understand the underlying algorithms; they simply input their data and receive actionable insights.

---

## Future Enhancements

- Add **real-time sensor integration** for automatic water quality monitoring.
- Expand the model to predict **specific contaminants**.
- Provide **geospatial analysis** of water quality across regions.

---

> Safe water is a human right. This project makes it **intelligent, fast, and accessible**.