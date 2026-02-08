import gradio as gr
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from utils import FeatureEngineer
from utils import IQRClipper

# 1. Import the trained model
model = joblib.load("stacking_pipeline_water.joblib")


# 2. Prediction function
def predict_potability(
    ph, hardness, solids, chloramines, sulfate, 
    conductivity, organic_carbon, trihalomethanes, turbidity
):
    input_df = pd.DataFrame([[
        ph, hardness, solids, chloramines, sulfate, 
        conductivity, organic_carbon, trihalomethanes, turbidity
    ]],
    columns=[
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ])
    
    pred_class = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0,1]
    
    result = "Potable" if pred_class == 1 else "Not Potable"
    return f"{result}"


# 3. Gradio sliders for user input
inputs = [
    gr.Slider(0, 14, step=0.1, value=7.0, label="pH"),
    gr.Slider(47, 323, step=1, value=200, label="Hardness"),
    gr.Slider(321, 61227, step=100, value=20000, label="Solids"),
    gr.Slider(0.35, 13.13, step=0.01, value=7.0, label="Chloramines"),
    gr.Slider(129, 481, step=1, value=300, label="Sulfate"),
    gr.Slider(181, 753, step=1, value=400, label="Conductivity"),
    gr.Slider(2.2, 28.3, step=0.1, value=15, label="Organic Carbon"),
    gr.Slider(0.74, 124, step=0.1, value=60, label="Trihalomethanes"),
    gr.Slider(1.45, 6.74, step=0.01, value=4, label="Turbidity"),
]


# 4. Launch Gradio interface
app = gr.Interface(
    fn=predict_potability,
    inputs=inputs,
    outputs="text",
    title="Water Potability Predictor",
    description="Predict if water is safe to drink based on chemical properties."
)

app.launch(share=True)