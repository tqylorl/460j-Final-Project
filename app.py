import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from datetime import datetime
from datasets import load_dataset
import plotly.express as px
import plotly.graph_objects as go
import time

# project imports
from config import MODELS_DIR, CALORIES_CSV, DEFAULT_PORTION_SIZE
from utils import load_image, normalize_food_name, calculate_calories, save_results, create_visualization

# Page configuration with custom theme
st.set_page_config(
    page_title="FoodVision AI",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        background: -webkit-linear-gradient(left, #FF5F6D, #FFC371);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #888;
        margin-top: 0;
    }
    .stButton>button {
        background-color: #FF5F6D;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FFC371;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .food-image {
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    .calorie-info {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .category-badge {
        background-color: #007bff;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 10px;
    }
    .dashboard-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for app controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/kawaii-french-fries.png", width=80)
    st.markdown("## 🍽️ FoodVision Controls")
    st.divider()
    
    # App mode selection
    app_mode = st.radio(
        "Choose App Mode",
        ["🔍 Food Analysis", "📊 Nutrition Dashboard", "ℹ️ About"]
    )
    
    # Parameters in sidebar
    if app_mode == "🔍 Food Analysis":
        st.subheader("Analysis Settings")
        portion = st.slider("Portion Size (g)", 
                            min_value=25, 
                            max_value=500, 
                            value=DEFAULT_PORTION_SIZE,
                            step=25,
                            help="Adjust the food portion size to estimate calories")
        
        top_k = st.slider("Number of Predictions", 
                         min_value=1, 
                         max_value=5, 
                         value=3,
                         help="How many food predictions to show")
                         
        show_confidence = st.toggle("Show Confidence Bars", 
                                   value=True,
                                   help="Display confidence score visualization")
    
    st.divider()
    st.markdown("##### Made with ❤️ by Team FoodVision")

# Cache functions for better performance
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
    # Get image sizes from config or model requirements
    target_size = (224, 224)
    if hasattr(load_keras_model(), 'input_shape'):
        input_shape = load_keras_model().input_shape
        if input_shape and len(input_shape) == 4:
            height, width = input_shape[1], input_shape[2]
            if height and width:  # If dimensions are not None
                target_size = (height, width)
    
    # Process the image
    pil_img, img_array = load_image(uploaded_file, target_size=target_size)
    return pil_img, np.expand_dims(img_array, axis=0)

def predict_food_keras(model, img_batch, class_names, top_k=3):
    with st.spinner("🔎 Analyzing your food..."):
        # Add small delay for UI effect
        time.sleep(0.5)
        probs = model.predict(img_batch)[0]
        top_idxs = np.argsort(probs)[-top_k:][::-1]
        results = [(class_names[i], float(probs[i])) for i in top_idxs]
        time.sleep(0.5)
    return results

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

def create_donut_chart(calories, protein=15, carbs=50, fat=35):
    # Create a sample macronutrient breakdown based on calories
    # In a real app, you would get this from a nutrition database
    total_grams = calories / 4  # rough estimate
    
    # Default macronutrient percentages (protein/carbs/fat)
    protein_g = (calories * (protein/100)) / 4
    carbs_g = (calories * (carbs/100)) / 4
    fat_g = (calories * (fat/100)) / 9
    
    labels = ['Protein', 'Carbohydrates', 'Fat']
    values = [protein_g, carbs_g, fat_g]
    colors = ['#FF5F6D', '#FFC371', '#38b6ff']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker=dict(colors=colors)
    )])
    
    fig.update_layout(
        showlegend=True,
        margin=dict(t=0, b=0, l=0, r=0),
        height=300,
        annotations=[dict(text=f'{calories:.0f}<br>kcal', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig

def create_confidence_chart(predictions):
    labels = [p[0].replace('_', ' ').title() for p in predictions]
    values = [p[1] * 100 for p in predictions]
    
    fig = px.bar(
        x=values, 
        y=labels, 
        orientation='h',
        text=[f"{v:.1f}%" for v in values],
        color=values,
        color_continuous_scale=['#FFC371', '#FF5F6D'],
        labels={'x': 'Confidence (%)', 'y': 'Food Item'}
    )
    
    fig.update_layout(
        xaxis_range=[0, 100],
        margin=dict(t=0, b=0, l=0, r=0),
        height=300,
        xaxis_title="",
        yaxis_title="",
        coloraxis_showscale=False
    )
    
    return fig

# Main Content Based on App Mode
if app_mode == "ℹ️ About":
    st.markdown("<h1 class='main-header'>FoodVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Intelligent Food Recognition & Nutrition Analysis</p>", unsafe_allow_html=True)
    
    st.write("---")
    
    st.markdown("""
    ### 🚀 Welcome to FoodVision AI!
    
    FoodVision is an advanced AI-powered food recognition system that helps you:
    
    - **Identify food items** from images with high accuracy
    - **Estimate calorie content** of your meals
    - **Track nutritional information** for better dietary choices
    
    ### 🔍 How It Works
    
    1. Upload a photo of your food
    2. Our AI model (trained on Food-101 dataset) identifies what you're eating
    3. We estimate calories and nutritional content based on the identified food
    4. You get instant nutrition insights!
    
    ### 💻 Technical Features
    
    - Deep learning model built with TensorFlow
    - Trained on 101,000 food images across 101 categories
    - Nutritional database with detailed calorie information
    
    Try it now by selecting "Food Analysis" mode in the sidebar!
    """)

elif app_mode == "📊 Nutrition Dashboard":
    st.markdown("<h1 class='main-header'>Nutrition Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Explore nutritional insights</p>", unsafe_allow_html=True)
    
    st.info("💡 Upload a food image in the Food Analysis mode first to see your personalized dashboard.")
    
    # Demo dashboard with placeholder data
    st.subheader("Sample Food Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://images.unsplash.com/photo-1513104890138-7c749659a591", 
                caption="Pizza Margherita", use_container_width=True)
    
    with col2:
        st.plotly_chart(create_donut_chart(285), use_container_width=True)
    
    st.subheader("Common Food Calorie Comparison")
    
    # Sample data for demonstration
    sample_foods = {
        'Pizza (1 slice)': 285,
        'Hamburger': 250, 
        'Caesar Salad': 150,
        'French Fries': 365,
        'Ice Cream (1 scoop)': 140,
        'Sushi Roll': 255,
        'Apple': 72
    }
    
    fig = px.bar(
        x=list(sample_foods.keys()),
        y=list(sample_foods.values()),
        labels={'x': 'Food Item', 'y': 'Calories (kcal)'},
        color=list(sample_foods.values()),
        color_continuous_scale=['#38b6ff', '#FF5F6D'],
        text=[f"{v} kcal" for v in sample_foods.values()]
    )
    
    fig.update_layout(
        xaxis_title="",
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:  # Food Analysis mode (default)
    # Main header
    st.markdown("<h1 class='main-header'>FoodVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Upload a food image to identify it and estimate calories</p>", unsafe_allow_html=True)
    
    # File uploader centered in the page
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    # If image is uploaded
    if uploaded:
        try:
            with st.spinner("Loading necessary components..."):
                class_names = load_class_names()
                model = load_keras_model()
                calorie_df = load_calorie_data()
            
            # Process the image and get predictions
            pil_img, batch = preprocess_for_keras(uploaded)
            
            # Layout for results
            col1, col2 = st.columns([6, 4])
            
            with col1:
                # Display the image with enhanced styling
                st.markdown("### Your Food")
                st.image(pil_img, use_container_width=True, clamp=True, 
                         output_format="PNG", channels="RGB", 
                         caption="Uploaded Image")
            
            with col2:
                # Get predictions with spinner animation
                top_k = top_k if 'top_k' in locals() else 3
                preds = predict_food_keras(model, batch, class_names, top_k=top_k)
                
                # Show predictions with styled cards
                st.markdown("### Analysis Results")
                
                # Get the top prediction
                best_label = preds[0][0]
                best_prob = preds[0][1]
                
                # Get calorie info
                calorie_row = find_calorie_info(best_label, calorie_df)
                cals100 = calorie_row.get('Cals_per100grams', np.nan)
                portion = portion if 'portion' in locals() else DEFAULT_PORTION_SIZE
                total_cals = calculate_calories(cals100 or 0, portion)
                
                # Top prediction card
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>📸 {best_label.replace('_', ' ').title()}</h3>
                    <span class="category-badge">{calorie_row.get('FoodCategory', 'Food')}</span>
                    <p>Confidence: <b>{best_prob:.1%}</b></p>
                    <div class="calorie-info">
                        <h4>🔥 Calorie Information</h4>
                        <p>Per 100g: <b>{cals100:.0f} kcal</b></p>
                        <p>Your portion ({portion}g): <b>{total_cals:.0f} kcal</b></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a macronutrient donut chart
                st.markdown("### Estimated Macronutrients")
                st.plotly_chart(create_donut_chart(total_cals), use_container_width=True)
            
            # Show confidence visualization if enabled
            show_confidence = show_confidence if 'show_confidence' in locals() else True
            if show_confidence:
                st.markdown("### Confidence Scores")
                conf_chart = create_confidence_chart(preds)
                st.plotly_chart(conf_chart, use_container_width=True)
            
            # Save button centered at the bottom
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("💾 Save Results", use_container_width=True):
                    with st.spinner("Saving your results..."):
                        success, path = save_results(
                            pil_img, 
                            {"food_name": best_label, "confidence": best_prob},
                            {"calories_per_100g": cals100, "portion_size_grams": portion, "total_calories": total_cals}
                        )
                        if success:
                            st.success(f"Results saved successfully in {path}")
                            time.sleep(1)
                            st.balloons()
                        else:
                            st.error("Failed to save results. Please try again.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Make sure the model is properly trained and all files are available.")
    
    else:
        # Display a helpful message when no image is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 50px 0;">
            <img src="https://img.icons8.com/fluency/96/000000/compact-camera.png" style="margin-bottom: 20px;">
            <h2>Upload a food image to begin</h2>
            <p>Supported formats: JPG, PNG, JPEG</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample food images
        st.markdown("### 🍕 Sample Food Categories")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image("https://images.unsplash.com/photo-1565299624946-b28f40a0ae38", caption="Pizza", use_container_width=True)
        with col2:
            st.image("https://images.unsplash.com/photo-1546069901-ba9599a7e63c", caption="Salad", use_container_width=True)
        with col3:
            st.image("https://images.unsplash.com/photo-1563379926898-05f4575a45d8", caption="Sushi", use_container_width=True)
        with col4:
            st.image("https://images.unsplash.com/photo-1586190848861-99aa4a171e90", caption="Burger", use_container_width=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #888;">
    <p>© 2023 FoodVision AI - Food Recognition & Calorie Estimation System</p>
</div>
""", unsafe_allow_html=True)
