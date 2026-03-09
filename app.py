
import pandas as pd
import joblib
import gradio as gr

# Load the best performing model and feature names
loaded_model = joblib.load('best_ada_model.joblib')
loaded_feature_names = joblib.load('feature_names.joblib')
loaded_label_encoders = joblib.load('label_encoders.joblib') # Load the saved encoders
loaded_scaler = joblib.load('scaler.joblib') # Load the saved scaler

def predict_heart_disease(
    Smoking: str,
    Age: int,
    Family_Heart_Disease: str,
    BMI: float,
    Cholesterol_Level: int,
    Blood_Pressure: int,
    Stress_Level: str,
    Diabetes: str,
    Homocysteine_Level: float
) -> tuple:
    """
    Predicts heart disease status based on input features using a pre-trained model.

    Args:
        Smoking (str): 'Yes' or 'No'
        Age (int): Patient's age
        Family_Heart_Disease (str): 'Yes' or 'No'
        BMI (float): Body Mass Index
        Cholesterol_Level (int): Cholesterol level
        Blood_Pressure (int): Blood pressure
        Stress_Level (str): 'High', 'Low', or 'Medium'
        Diabetes (str): 'Yes' or 'No'
        Homocysteine_Level (float): Homocysteine level

    Returns:
        tuple: A tuple containing (pd.DataFrame of input features, str prediction result)
    """

    # Prepare a dictionary for the input values, matching the expected column names
    input_raw_data = {
        'Smoking': Smoking,
        'Age': Age,
        'Family Heart Disease': Family_Heart_Disease,
        'BMI': BMI,
        'Cholesterol Level': Cholesterol_Level,
        'Blood Pressure': Blood_Pressure,
        'Stress Level': Stress_Level,
        'Diabetes': Diabetes,
        'Homocysteine Level': Homocysteine_Level
    }

    # Create a DataFrame from the single row of input data
    input_df_processed = pd.DataFrame([input_raw_data])

    # Identify categorical and numerical columns based on `loaded_feature_names` and `loaded_label_encoders`
    cat_cols_to_encode = [col for col in loaded_label_encoders.keys() if col in input_df_processed.columns]
    num_cols_to_scale = [col for col in loaded_feature_names if col not in loaded_label_encoders.keys()]

    # Encode categorical features using the loaded encoders
    for col in cat_cols_to_encode:
        encoder = loaded_label_encoders[col]
        input_df_processed[col] = encoder.transform(input_df_processed[col])

    # Select and order features according to the `loaded_feature_names`
    input_df = input_df_processed[loaded_feature_names]

    # Apply scaler to numerical features
    input_df[num_cols_to_scale] = loaded_scaler.transform(input_df[num_cols_to_scale])

    # Use the loaded_model to make a prediction
    prediction = loaded_model.predict(input_df)[0]

    # Convert the numerical prediction back into a user-friendly string
    if prediction == 1:
        result = 'Heart Disease Detected'
    else:
        result = 'No Heart Disease'

    # Return this user-friendly prediction string and the input DataFrame
    return input_df, result

# Define input components based on the predict_heart_disease function's parameters
inputs = [
    gr.Dropdown(['Yes', 'No'], label="Smoking"),
    gr.Slider(minimum=1, maximum=90, step=1, label="Age"),
    gr.Dropdown(['Yes', 'No'], label="Family Heart Disease"),
    gr.Slider(minimum=18.0, maximum=35.0, step=0.1, label="BMI"),
    gr.Slider(minimum=150, maximum=319, step=1, label="Cholesterol Level"),
    gr.Slider(minimum=90, maximum=179, step=1, label="Blood Pressure"),
    gr.Dropdown(['High', 'Medium', 'Low'], label="Stress Level"),
    gr.Dropdown(['Yes', 'No'], label="Diabetes"),
    gr.Slider(minimum=5.0, maximum=19.99, step=0.01, label="Homocysteine Level")
]

# Define output components
output_features_table = gr.DataFrame(headers=[col.replace('_', ' ').title() for col in loaded_feature_names], label="Input Features", row_count=(1, "fixed"), col_count=(len(loaded_feature_names), "fixed"), interactive=False)
output_prediction = gr.Label(label="Prediction Result")

outputs = [output_features_table, output_prediction]

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=inputs,
    outputs=outputs,
    title="Heart Disease Prediction",
    description="Enter patient details to predict the likelihood of heart disease.",
    clear_btn="Clear Outputs"
)

# This part is for running the app.py when deployed, not for Colab directly
# if __name__ == "__main__":
#     iface.launch(share=True)
