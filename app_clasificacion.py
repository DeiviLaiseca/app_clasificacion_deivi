import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title('Course Grade Prediction') # Changed title

# Load the trained models and encoders
try:
    onehotencoder = joblib.load('onehot_encoder.pkl')
    standard_scaler = joblib.load('minmax_scaler.pkl')
    # Assuming you have a regression model saved as 'best_regression_model.pkl'
    prediction_model = joblib.load('best_model.pkl') # Assuming best_model.pkl is the regression model
except FileNotFoundError:
    st.error("Error: Model files not found. Please make sure 'onehot_encoder.pkl', 'minmax_scaler.pkl', and 'best_model.pkl' are in the same directory.")
    st.stop()

st.write("Please provide the following information to predict the course grade.") # Changed text

# Get user inputs
felder_options = ['activo', 'visual', 'equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal']
selected_felder = st.selectbox('Felder Learning Style', felder_options)
examen_admision = st.number_input('University Admission Exam Score', min_value=0.0, max_value=5.0, step=0.01)

# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    'Felder': [selected_felder],
    'Examen_admisión_Universidad': [examen_admision]
})

# Preprocess the input data
# Scale the numerical feature
input_data['Examen_admisión_Universidad'] = standard_scaler.transform(input_data[['Examen_admisión_Universidad']])

# Apply one-hot encoding to the 'Felder' column
try:
    felder_encoded = onehotencoder.transform(input_data[['Felder']])
    feature_names = onehotencoder.get_feature_names_out(['Felder'])
except AttributeError:
    feature_names = [f'Felder_{cat}' for cat in onehotencoder.categories_[0]]

felder_encoded_df = pd.DataFrame(felder_encoded, columns=feature_names, index=input_data.index)

# Concatenate the encoded features with the original DataFrame (excluding the original 'Felder' column)
processed_input_data = pd.concat([input_data.drop('Felder', axis=1), felder_encoded_df], axis=1)

# Rename the scaled column to match the training data column name
processed_input_data = processed_input_data.rename(columns={'Examen_admisión_Universidad': 'Examen_admisión_Universidad_scaled'})

# Ensure the columns are in the same order as the training data
# Assuming the training data columns order can be retrieved from the loaded model or a saved list
# For this example, we'll manually list the expected order based on previous steps
expected_columns = ['Felder_activo', 'Felder_equilibrio', 'Felder_intuitivo', 'Felder_reflexivo',
                    'Felder_secuencial', 'Felder_sensorial', 'Felder_verbal', 'Felder_visual',
                    'Examen_admisión_Universidad_scaled']

processed_input_data = processed_input_data.reindex(columns=expected_columns, fill_value=0)


# Make prediction
if st.button('Predict Grade'):
    # Realizar la predicción
    prediction = prediction_model.predict(processed_input_data)

    # 🔍 Depuración opcional (puedes comentar estas líneas después)
    st.write("Predicción cruda:", prediction)
    st.write("Tipo:", type(prediction))

    # Extraer el valor real, sin importar el formato
    try:
        predicted_value = float(np.array(prediction).flatten()[0])
        st.subheader('Predicted Grade')
        st.markdown(f"<h3 style='text-align:center; color:green;'>{predicted_value:.2f}</h3>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error al procesar la predicción: {e}")
        st.stop()
