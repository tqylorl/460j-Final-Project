# Food Calorie Estimator Frontend

This directory contains the frontend application for the Food Calorie Estimator project.

## Structure

- `streamlit/` - Contains the Streamlit web application
  - `streamlit_app.py` - Main Streamlit application file
- `utils.py` - Utility functions for the application
- `config.py` - Configuration settings
- `calories.csv` - Database of food items and their calorie information

## Setup and Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run streamlit/streamlit_app.py
   ```

## Features

- Upload food images for classification
- Get calorie estimates based on food type and portion size
- Save and visualize results 