# About:
- Creating a machine learning model **Logistic Regression** to predect a person's weight type wheather it's one of these categories `Insufficient Weight`, `Normal Weight`,
`Obesity Type I`, `Obesity Type II`, `Obesity Type III`, `Overweight Level I`, `Overweight Level II`.
# Steps:
- Preprocessing **ObesityDataSet.csv** by cleaning Na values, encoding categorical values and scaling numerical values.
- Performing **EDA** on the data to gain insights.
- Training a **Logistic Regression** model and saving it via **pickle** library with accuracy of **74%**.
# Files:
- **notebook.ipynb:** a notebook representing the **experimants** conducted to get from A to B.
- **predection.py:** the streamlit app to make the predection process seamless and easy for the user.
- **models folder:** the folder for the **saved models**, right now we have the Logistic Regression model only.
- **encoders folder:** the folder for the saved encoders like **one hot**, **ordinal** and **min max scaler**. So we can use them in the app.
- **utils.py:** a helper python file for visualzations, outlier detection and missing values display.
- **requirements.txt:** a file to get the nessecery **liberaries** into the streamlit app.
# How to Run:
- **First Method:** you can go here to see the app right away https://obesitydetection-89jtamzebuqdk8bcffpbhr.streamlit.app/
- **Second Method:** download the repository and type
```bash
# Install dependencies
pip install -r requirements.txt  

# Run the app (replace with your path (your directory/predection.py) if not in the same directory)
streamlit run predection.py
