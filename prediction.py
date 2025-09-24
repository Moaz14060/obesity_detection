import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder

# Customizing the page
st.set_page_config(page_title="Predictions", page_icon=":material/online_prediction:")

# Main Title
st.markdown("<h1 style='text-align: center; color: black;'>Welcome</h1>", unsafe_allow_html=True)

# Subtitle
st.header("In this App:")
# Explaining the app
st.write("**First,** Preprocessing data that consists the estimation of obesity levels in people" \
" from the countries of Mexico, Peru and Colombia, with ages between 14 and 61 and diverse eating habits and physical condition.")
st.write("**Second,** exploring it visaualy using **EDA**.")
st.write("**Third,** training a **logistic regression** model with accuracy of **74%**.")
st.write("**Finally,** Predecting **weight category** using an attractive interface.")
st.divider()

# Radio buttons for gender feature
gender = st.radio("Select your gender: ", ("Male", "Female"))

# Slider for age feature
age = st.slider("What is your age: ", 16, 70, 20)

# Slider for height feature
height = st.slider("Select your height: ", 1.45, 2.00, 1.6)

# Slider for weight feature
weight = st.slider("Select your weight: ", 40, 180, 70)

# Radio buttons for family history with overweight
famliy_history = st.radio("Does your family have history with overweight ?", ("Yes", "No"))
# Because the values of the feature is in lower case format
famliy_history = famliy_history.lower()

# Radio buttons for frequent consumption of high caloric foods feature
frequent_cons_high_cal_food = st.radio("Do you frequently eat foods that has high calories ?", ("Yes", "No"))
# Because the values of the feature is in lower case format
frequent_cons_high_cal_food = frequent_cons_high_cal_food.lower()

# Slider for vegetables consumption freqency feature
vegetables = st.slider("How often do you eat vegetables: ", 1, 3, 1)

# Slider for how many meals per day feature
meals = st.slider("How many meals do you have a day: ", 1, 4, 3)

# Select box for the frequent consumption of snacks between meals feature
food_between = st.selectbox("How often do you snacks between meals: ", options=("Always", "Frequently", "Sometimes", "no".title()))

# Radio buttons for whether the user smokes or not
smoke = st.radio("Do you smoke ?", ("Yes", "No"))
# Because the values of the feature is in lower case format
smoke = smoke.lower()

# Slider for the average consumption of water in liters feature
water = st.slider("what is the average liters of water you consume per day: ", 0.5, 3.00, 1.5)

# Radio button for wheather the user monitors their calories intake feature
calory_monitor = st.radio("Do you monitor your consumption of calories ?", ("Yes", "No"))
# Because the values of the feature is in lower case format
calory_monitor = calory_monitor.lower()

# Slider for how many times the user exercise in week 
physical = st.slider("How many times do you exercise per week: ", 0, 3, 2)

# Slider for average hours  spent on technology  
tech = st.slider("What is the average hours you spend on technology devices: ", 0, 4, 2)

# Select box for the frequency of alcohol consumption feature
alcohol = st.selectbox("How often do you drink alcohol: ", options=("Always", "Frequently", "Sometimes", "no".title()))

# Select box for what type of method for transportation feature
transport = st.selectbox("What method do you use the most for transportation: ", options=("Public_Transportation".replace("_", " "), "Automobile", 
                                                                                          "Walking", "Motorbike", "Bike"))


# The value for predection
value = {
    "Gender": gender, 
    "Age": age, 
    "Height": height, 
    "Weight": weight, 
    "family_history_with_overweight": famliy_history, 
    "frequent_consumption_of_high_caloric_food": frequent_cons_high_cal_food, 
    "frequency_of_consumption_of_vegetables": vegetables, 
    "number_of_main_meals": meals, 
    "consumption_of_food_between_meals": food_between, 
    "SMOKE": smoke, 
    "consumption_of_water_daily": water, 
    "calories_consumption_monitoring": calory_monitor, 
    "physical_activity_frequency": physical, 
    "time_using_technology_devices": tech, 
    "consumption_of_alcohol": alcohol, 
    "transportation_used": transport 
}

@st.cache_data
def preprocess (value: dict, _one_hot: OneHotEncoder, _min_max_scaler: MinMaxScaler, _ordinal_encoder: OrdinalEncoder) -> pd.DataFrame:
    """
        The function transforms the entered value into a data frame then passes it thruogh ordinal_encoding, one_hot_encoding
        and min_max_scaling and returns a preprocessed data frame

        Parameters:
            1. value: the values of the features we want to predict (dict)
            2. one_hot: an object of the OneHotEncoder (OneHotEncoder) 
            3. min_max_scaler: an object of the MinMaxScaler (MinMaxScaler)
            4. ordinal_encoder: an object of the OrdinalEncoder (OrdinalEncoder)

        Return Value:
            the preprocessed values as a data frame
    """
    cate_for_one_hot_encoder = ["consumption_of_food_between_meals", 'consumption_of_alcohol', 'transportation_used']

    cate_for_label_encoder=["Gender", "family_history_with_overweight",
                        "frequent_consumption_of_high_caloric_food", "SMOKE",
                        "calories_consumption_monitoring"]
    
    numer_col = ["Age", "Height", "Weight", "frequency_of_consumption_of_vegetables", "number_of_main_meals", 
             "consumption_of_water_daily", "physical_activity_frequency", "time_using_technology_devices"]
    value_df = pd.DataFrame([value])
    # Encoding categorical features with only two categories
    value_df[cate_for_label_encoder]=ordinal_encoder.transform(value_df[cate_for_label_encoder])
    # One hot encoding
    encoded_value= one_hot.transform(value_df[cate_for_one_hot_encoder])
    encoded_value_df = pd.DataFrame(encoded_value, columns=one_hot.get_feature_names_out(cate_for_one_hot_encoder), index= value_df.index)
    final_value_df = pd.concat([encoded_value_df, value_df.drop(columns=cate_for_one_hot_encoder)], axis=1)
    # Min max scaling
    final_value_df[numer_col] = min_max_scaler.transform(final_value_df[numer_col])

    return final_value_df

# Load Encoders
with open("encoders/one_hot.pkl", "rb") as f:
    one_hot = pickle.load(f)

with open("encoders/scaler.pkl", "rb") as f:
    min_max_scaler = pickle.load(f)

with open("encoders/ordinal.pkl", "rb") as f:
    ordinal_encoder = pickle.load(f)

# Preprocess the value
value_to_pred = preprocess(value, one_hot, min_max_scaler, ordinal_encoder)

# Load Logistic Regression Model
with open("models/logistic_model.sav", "rb") as f:
    log_model = pickle.load(f)

# A method to center the button
col1, col2, col3, col4, col5 = st.columns(5)
with col3:
    pred = st.button("Predict", type= "primary", icon = ":material/online_prediction:")
if pred:
    weight_type = log_model.predict(value_to_pred)[0].replace("_", " ")
    if weight_type == "Normal Weight":
        st.success(f"Your Weight Type: **{weight_type}**")
    elif weight_type in ["Overweight Level I", "Overweight Level II", "Insufficient Weight"]:
        st.warning(f"Your Weight Type: **{weight_type}**")
        st.warning("Be Cautious!")
    else:
        st.error(f"Your Weight Type: **{weight_type}**")

        st.warning("Go See a Doctor Immediatley!")

