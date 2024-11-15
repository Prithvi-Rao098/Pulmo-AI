import gradio as gr
import pandas as pd
import os
from joblib import load

# Load the trained model
model = load("../models/param_predict_model.joblib")

# Define column names as per model training
column_names = [
    'air_pollution', 'alcohol_use', 'dust_allergy', 'occupational_hazards', 'genetic_risk',
    'chronic_lung_disease', 'balanced_diet', 'obesity', 'smoking', 'passive_smoker',
    'chest_pain', 'coughing_of_blood', 'fatigue', 'weight_loss', 'shortness_of_breath',
    'wheezing', 'swallowing_difficulty', 'clubbing_of_finger_nails', 'frequent_cold',
    'dry_cough', 'snoring'
]

# Prediction function for lung cancer risk level
def predict_lung_cancer(
    air_pollution, alcohol_use, dust_allergy, occupational_hazards, genetic_risk,
    chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoker,
    chest_pain, coughing_of_blood, fatigue, weight_loss, shortness_of_breath,
    wheezing, swallowing_difficulty, clubbing_of_finger_nails, frequent_cold,
    dry_cough, snoring
):
    # Combine features into a DataFrame
    input_data = [air_pollution, alcohol_use, dust_allergy, occupational_hazards, genetic_risk,
                  chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoker,
                  chest_pain, coughing_of_blood, fatigue, weight_loss, shortness_of_breath,
                  wheezing, swallowing_difficulty, clubbing_of_finger_nails, frequent_cold,
                  dry_cough, snoring]
    input_df = pd.DataFrame([input_data], columns=column_names)
    
    # Predict probabilities for each risk level
    probabilities = model.predict_proba(input_df)[0]
    result = {
        "Low Risk (Class 0)": f"{probabilities[0] * 100:.2f}%",
        "Medium Risk (Class 1)": f"{probabilities[1] * 100:.2f}%",
        "High Risk (Class 2)": f"{probabilities[2] * 100:.2f}%"
    }
    return result

# Define inputs using Gradio components
inputs = [gr.Slider(1, 9, step=1, label=name.replace('_', ' ').title()) for name in column_names]

# Define output as a labeled display of the prediction
outputs = gr.JSON(label="Lung Cancer Risk Prediction")

# Create the Gradio Interface
app = gr.Interface(
    fn=predict_lung_cancer,
    inputs=inputs,
    outputs=outputs,
    title="Lung Cancer Risk Prediction",
    description="Adjust the sliders for each factor to predict the lung cancer risk level."
)

# Launch the app with network settings and optional proxy configuration
proxy_prefix = os.environ.get("PROXY_PREFIX")
app.launch(server_name="0.0.0.0", server_port=8080, root_path=proxy_prefix)