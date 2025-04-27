import pandas as pd
from food_classifier import classify_food

from fuzzywuzzy import fuzz

# load calories data
calories_df = pd.read_csv('data/calories.csv')

# estimate calories based on food prediction
def estimate_calories(predicted_food, weight_in_grams):
    # find food in dataframe
    food_row = calories_df[calories_df['FoodItem'] == predicted_food]

    if not food_row.empty:
        # get calories per 100 grams
        cals_per_100g = food_row['Cals_per100grams'].values[0]

        # calculate cals baesd on weight
        estimated_calories = (cals_per_100g * weight_in_grams) / 100
        return estimated_calories
    else:
        return None # predicted category not in csv

# use fuzzing to get the closest match in the data if we don't have an exact match
def get_closest_food(predicted_food):
    closest_match = max(calories_df['FoodItem'], key=lambda x:fuzz.ratio(predicted_food.lower(), x.lower()))

    return closest_match
    
def classify_and_estimate(image_path, weight_in_grams=100):
    # predict which food
    predicted_food = classify_food(image_path)

    # estimate calories
    calories = estimate_calories(predicted_food, weight_in_grams)

    # if exact match  not found in data
    if calories is None:
        # get closest match
        closest_food = get_closest_food(predicted_food)
        print(f"predicted food: '{predicted_food}' not found. using closest match: {closest_food}")
        calories = estimate_calories(closest_food, weight_in_grams)
    
    if calories is not None:
        print(f"predicted food: {predicted_food}")
        print(f"estimated calories for {weight_in_grams} grams: {calories:.2f} kcal")
    else:
        print(f"unable to estimate calories for {predicted_food} or closest match")

image_path = input("enter image path: ")
food_weight = int(input("enter food weight (grams): "))

classify_and_estimate(image_path, weight_in_grams=food_weight)
        
