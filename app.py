import pickle
import pandas as pd
import numpy as np
import streamlit as st

def predict_species(sep_len, sep_width, pet_len, pet_width,scaler_path,model_path):
   try:
      # Load the scaler 
      with open(scaler_path, 'rb') as file1:
         scaler = pickle.load(file1)
      with open(model_path, 'rb') as file2:
         model = pickle.load(file2)

      dict = {'SepalLengthCm':[sep_len], 'SepalWidthCm':[sep_width], 'PetalLengthCm':[pet_len], 'PetalWidthCm':[pet_width]}

      x_new = pd.DataFrame(dict)

      xnew_pre = scaler.transform(x_new)
     
      pred = model.predict(xnew_pre)
      prob = model.predict_proba(xnew_pre)
      max_prob = np.max(prob)
      return pred, max_prob
   except Exception as e:
      st.error(f"Error during prediction: {str(e)}")
      return None, None
    
st.title("Iris Species Predictor")

sep_len = st.number_input("Sepal Length", min_value=0.0, value=5.1, step=0.1)
sep_width = st.number_input("Sepal Width", min_value=0.0, value=3.5, step=0.1)
pet_len = st.number_input("Petal Length", min_value=0.0, value=1.4, step=0.1)
pet_width = st.number_input("Petal Width", min_value=0.0, value=3.5, step=0.1)

if st.button("Predict"):
   scaler_path = 'notebook/scaler.pkl'
   model_path = 'notebook/model.pkl'
   pred, max_prob = predict_species(sep_len, sep_width, pet_len, pet_width, scaler_path, model_path)

   if pred is not None and max_prob is not None:
      st.subheader(f"Predicted Species: {pred[0]}")
      st.subheader(f"Prediction Probability: {max_prob:.4f}")
   else:
      st.error("Prediction failed. Please check the input values are model files and try again.")