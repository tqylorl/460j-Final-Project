import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from datetime import datetime
from datasets import load_dataset
import sys

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# project imports
from config import MODELS_DIR, CALORIES_CSV, DEFAULT_PORTION_SIZE
from utils import load_image, normalize_food_name, calculate_calories, save_results, create_visualization

st.set_page_config(
    page_title="Food Image Calorie Estimator",
    page_icon="🍎",
    layout="wide"
)

st.title("Food Image Calorie Estimator")
st.markdown("""
Upload a food image to classify the dish and estimate its caloric content.
This app uses your Keras EfficientNetB0 model and a simple CSV lookup for calories.
""")

@st.cache_resource
def load_class_names():
    ds = load_dataset("ethz/food101", split="train[:1]")  # small slice just to get names
    return ds.features["label"].names

@st.cache_resource
def load_keras_model():
    model_path = os.path.join(MODELS_DIR, "food_classifier_model.h5")
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_calorie_data():
    df = pd.read_csv(CALORIES_CSV)
    df['Cals_per100grams'] = pd.to_numeric(df['Cals_per100grams'], errors='coerce')
    return df

def preprocess_for_keras(uploaded_file):
    # returns (PIL Image, np.array of shape (1,224,224,3))
    pil_img, img_array = load_image(uploaded_file, target_size=(224,224))
    return pil_img, np.expand_dims(img_array, axis=0)

def predict_food_keras(model, img_batch, class_names, top_k=3):
    probs = model.predict(img_batch)[0]
    top_idxs = np.argsort(probs)[-top_k:][::-1]
    return [(class_names[i], float(probs[i])) for i in top_idxs]

def find_calorie_info(food_label, df):
    clean = food_label.lower().replace('_',' ')
    exact = df[df['FoodItem'].str.lower()==clean]
    if not exact.empty: 
        return exact.iloc[0]
    partial = df[df['FoodItem'].str.lower().str.contains(clean)]
    if not partial.empty: 
        return partial.iloc[0]
    return pd.Series({
      'FoodCategory':'Unknown',
      'FoodItem':food_label,
      'per100grams':'100g',
      'Cals_per100grams':np.nan,
      'KJ_per100grams':np.nan
    })

# Main UI
uploaded = st.file_uploader("Choose a food image", type=["jpg","png","jpeg"])
portion = st.number_input("Portion size (g)", min_value=1, value=DEFAULT_PORTION_SIZE)

if uploaded:
    class_names = load_class_names()
    model = load_keras_model()
    calorie_df = load_calorie_data()

    pil_img, batch = preprocess_for_keras(uploaded)
    preds = predict_food_keras(model, batch, class_names)
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    st.subheader("Top Predictions")
    for label, prob in preds:
        st.write(f"**{label.replace('_',' ').title()}** — {prob:.1%}")

    best_label = preds[0][0]
    calorie_row = find_calorie_info(best_label, calorie_df)
    cals100 = calorie_row.get('Cals_per100grams', np.nan)
    total_cals = calculate_calories(cals100 or 0, portion)

    st.subheader("Calorie Estimate")
    st.write(f"Food Item: **{calorie_row.get('FoodItem','Unknown').title()}**")
    st.write(f"Calories per 100 g: **{cals100:.0f}**")
    st.write(f"Portion: **{portion:.0f} g** → **{total_cals:.0f} kcal**")

    # optional: show bar chart of top‑k
    vis_buf = create_visualization(pil_img, ([i for i,_ in enumerate(preds)], [p for _,p in preds]))
    if vis_buf:
        st.image(vis_buf, caption="Confidence Scores", use_column_width=True)

    if st.button("Save Results"):
        success, path = save_results(pil_img, {"food_name":best_label,"confidence":preds[0][1]},
                                     {"calories_per_100g":cals100,"portion_size_grams":portion,"total_calories":total_cals})
        if success:
            st.success(f"Results saved in `{path}`")
