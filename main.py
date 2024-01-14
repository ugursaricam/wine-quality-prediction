import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

#st.set_page_config(layout="wide")

class FeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['acidity_ratio'] = X['fixed acidity'] / X['volatile acidity']
        X['free_sulfur/total_sulfur'] = X['free sulfur dioxide'] / X['total sulfur dioxide']
        X['sugar/alcohol'] = X['residual sugar'] / X['alcohol']
        X['alcohol/density'] = X['alcohol'] / X['density']
        X['total_acid'] = X['fixed acidity'] + X['volatile acidity'] + X['citric acid']
        X['sulphates/chlorides'] = X['sulphates'] / X['chlorides']
        X['bound_sulfur'] = X['total sulfur dioxide'] - X['free sulfur dioxide']
        X['alcohol/pH'] = X['alcohol'] / X['pH']
        X['alcohol/acidity'] = X['alcohol'] / X['total_acid']
        X['alkalinity'] = X['pH'] + X['alcohol']
        X['mineral'] = X['chlorides'] + X['sulphates'] + X['residual sugar']
        X['density/pH'] = X['density'] / X['pH']
        X['total_alcohol'] = X['alcohol'] + X['residual sugar']
        X['acid/density'] = X['total_acid'] / X['density']
        X['sulphate/density'] = X['sulphates'] / X['density']
        X['sulphates/acid'] = X['sulphates'] / X['volatile acidity']
        X['sulphates*alcohol'] = X['sulphates'] * X['alcohol']
        return X

@st.cache_data
def get_data():
    df = pd.read_csv("WineQT.csv")
    return df

@st.cache_data
def get_pipeline():
    pipeline = joblib.load("wine_quality.joblib")
    return pipeline


st.title("DSS - :red[Wine] Quality Prediction 🍷🍇")

main_page, data_page, model_page = st.tabs(["Main Page", "Dataset", "ML Model"])

# Main Page
main_page.image("image/wineqt3.png")
main_page.subheader("Introduction:")
main_page.markdown("Welcome to the Wine Quality Prediction App! This tool leverages advanced data analysis techniques to predict the quality of wine based on various physicochemical properties. Using the Ekteki dataset, we have developed a model that classifies wine quality into three categories: low, medium, and high.")
main_page.subheader("How It Works:")
main_page.markdown("Our model is built on a comprehensive dataset that includes several parameters like acidity, sugar content, alcohol level, and more. We have transformed the quality rating into a categorical variable with three levels: low (0), medium (1), and high (2), to enhance the prediction accuracy. This categorization is based on the original quality scores, where scores up to 4 are considered low, scores between 5 and 6 are medium, and scores above 6 are high.")
main_page.subheader("Model Performance:")
main_page.markdown("The prediction model has demonstrated high accuracy with a score of 0.8863. This indicates that our model is highly effective in predicting the quality category of wine based on its physicochemical characteristics.")


# Dataset
data_page.subheader("Dataset Overview:")
data_page.markdown("The dataset used in this Wine Quality Prediction App is a comprehensive collection of data on various wines. It primarily focuses on physicochemical properties and quality ratings, providing valuable insights into the factors that influence the quality of wine.")

data_page.subheader("Key Features:")
data_page.markdown("""
- Fixed Acidity: This measures the total amount of tartaric acid in the wine and is expressed in grams per liter. A higher fixed acidity often results in a more tart flavor.
- Volatile Acidity: This indicates the amount of acetic acid in the wine, too much of which can lead to an unpleasant vinegar taste.
- Citric Acid: Found in small quantities, citric acid can add freshness and flavor to wines.
- Residual Sugar: This refers to the amount of sugar remaining after fermentation stops. It's usually measured in grams per liter.
- Chlorides: The amount of salt in the wine.
- Free Sulfur Dioxide: The free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine.
- Total Sulfur Dioxide: The amount of free and bound forms of S02. Too high a concentration can make the wine smell like burnt matches.
- Density: The density of wine is close to that of water, depending on the alcohol and sugar content.
- pH: Describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale.
- Sulphates: A wine additive that can contribute to sulfur dioxide gas (S02) levels, which acts as an antimicrobial and antioxidant.
- Alcohol: The percentage of alcohol content in the wine.
- Quality: The output variable based on sensory data, scored between 0 (very poor) to 10 (excellent).
""")

df = get_data()
data_page.dataframe(df)

data_page.subheader("Dataset Usage:")
data_page.markdown("In our app, the quality of wine is predicted based on these features. The original quality scores have been transformed into three categories for the purposes of this model: low (0), medium (1), and high (2). This transformation aids in enhancing the predictive accuracy of our model.")

# Model Page

model_page.col1, model_page.col2 = model_page.columns([1,1])

pipeline = get_pipeline()

# Creating input fields for each parameter
fixed_acidity = model_page.col1.slider('Fixed Acidity', min_value=0.0, max_value=16.0, value=0.1, step=0.1)
volatile_acidity = model_page.col1.slider('Volatile Acidity', min_value=0.0, max_value=2.0, value=0.01, step=0.01)
citric_acid = model_page.col1.slider('Citric Acid', min_value=0.0, max_value=1.5, value=0.01, step=0.01)
residual_sugar = model_page.col1.slider('Residual Sugar', min_value=0.0, max_value=16.0, value=0.1, step=0.1)
chlorides = model_page.col1.slider('Chlorides', min_value=0.0, max_value=1.0, value=0.01, step=0.001)
free_sulfur_dioxide = model_page.col1.slider('Free Sulfur Dioxide', min_value=0.0, max_value=100.0, value=0.1, step=1.0)
total_sulfur_dioxide = model_page.col1.slider('Total Sulfur Dioxide', min_value=0.0, max_value=300.0, value=0.1, step=1.0)
density = model_page.col1.slider('Density', min_value=0.0, max_value=1.5, value=0.01, step=0.0001)
pH = model_page.col1.slider('pH', min_value=0.1, max_value=4.5, value=0.1, step=0.01)
sulphates = model_page.col1.slider('Sulphates', min_value=0.0, max_value=2.5, value=0.01, step=0.01)
alcohol = model_page.col1.slider('Alcohol', min_value=0.0, max_value=15.0, value=0.1, step=0.1)

data = {
    'fixed acidity': fixed_acidity,
    'volatile acidity': volatile_acidity,
    'citric acid': citric_acid,
    'residual sugar': residual_sugar,
    'chlorides': chlorides,
    'free sulfur dioxide': free_sulfur_dioxide,
    'total sulfur dioxide': total_sulfur_dioxide,
    'density': density,
    'pH': pH,
    'sulphates': sulphates,
    'alcohol': alcohol
}

sample_df = pd.DataFrame(data, index=[0])


if model_page.col2.button("Predict!"):
    prediction = pipeline.predict(sample_df)
    if prediction == [0]:
        model_page.col2.success(f"Predicted Wine Quality is Low")
        model_page.col2.image("image/low.png")
    elif prediction == [1]:
        model_page.col2.success(f"Predicted Wine Quality is Medium")
        model_page.col2.image("image/medium.png")
    else:
        model_page.col2.success(f"Predicted Wine Quality is High")
        model_page.col2.image("image/high.png")


