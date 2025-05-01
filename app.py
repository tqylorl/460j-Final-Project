import streamlit as st
import torch
from PIL import Image
import io

from food_classifier import classify_food
from calorie_estimator import estimate_calories

import os

def main():
    # Set page config first
    st.set_page_config(
        page_title="FoodVision AI",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for modern styling with dark mode support
    st.markdown("""
        <style>
        /* Base styles */
        .main {
            background-color: transparent;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 20px;
            padding: 10px 25px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        
        /* Card styling with dark mode support */
        [data-testid="stSidebar"] {
            background-color: rgba(0,0,0,0.1);
        }
        
        .upload-section {
            background-color: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metric-card {
            background-color: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            margin: 10px 0;
        }
        
        .metric-card h3 {
            color: #4CAF50;
            margin: 0;
            font-size: 24px;
        }
        
        .metric-card h4 {
            color: rgba(255,255,255,0.8);
            margin: 0 0 10px 0;
            font-size: 16px;
            font-weight: normal;
        }
        
        /* Dark mode specific adjustments */
        @media (prefers-color-scheme: dark) {
            .metric-card {
                background-color: rgba(0,0,0,0.2);
            }
            .metric-card h4 {
                color: rgba(255,255,255,0.7);
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Header with gradient background
    st.markdown("""
        <div style='background: linear-gradient(45deg, #4CAF50, #45a049);
                   padding: 20px;
                   border-radius: 10px;
                   margin-bottom: 20px;
                   box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h1 style='color: white; text-align: center; margin: 0;'>FoodVision AI</h1>
            <p style='color: white; text-align: center; margin: 5px 0 0 0;'>
                Smart Food Recognition
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar with info
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.info(
            "FoodVision AI uses advanced machine learning to identify food items "
            "in images. Simply upload an image to get started!"
        )
        
        st.markdown("### 📊 Features")
        st.markdown("""
        - 🖼️ Image Recognition
        - 📱 Mobile-friendly Design
        - 🔍 Real-time Analysis
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📸 Upload Food Image")
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a single food item"
        )
        
        if uploaded_file is not None:
            try:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Save the uploaded file temporarily
                temp_path = "temp_upload.jpg"
                image.save(temp_path)
                
                # Analyze button with loading animation
                if st.button("🔍 Analyze Food", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        try:
                            # Get food classification
                            food_type = classify_food(temp_path)

                            # get calories (assuming 200g for now)
                            cals = estimate_calories(food_type, 200)
                            
                            with col2:
                                st.markdown("### 📊 Results")
                                
                                # Food type card with improved styling
                                st.markdown(f"""
                                    <div class='metric-card'>
                                        <h4>Detected Food</h4>
                                        <h3>{food_type.replace('_', ' ').title()}</h3>
                                        <h4>Estimated Calories</h4>
                                        <h3>{cals}</h3>
                                        <h3>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                        finally:
                            # Clean up temporary file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # Footer with tips
    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use well-lit, clear images
        - Ensure the food item is clearly visible
        - For best results, use images with a single food item
        """)
    with col4:
        st.markdown("### ⚡ Quick Start")
        st.markdown("""
        1. Upload a food image
        2. Click 'Analyze Food'
        """)
    with col5:
        st.markdown("### 🔍 Best Practices")
        st.markdown("""
        - Center the food in the frame
        - Avoid shadows and glare
        - Use a plain background
        """)

if __name__ == "__main__":
    main() 
