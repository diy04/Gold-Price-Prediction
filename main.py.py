import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from PIL import Image
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Set up the main title and the header image
st.title('Gold Price Prediction')

# Display a background image (use the one you uploaded)
img = Image.open('image.jpeg')  # Ensure 'image.png' is the correct file path
st.image(img, use_column_width=True)

# Add a header for file upload functionality
st.subheader('Upload Gold Price Data')

# Allow the user to upload a CSV file
uploaded_file = st.file_uploader("Choose a file", type="csv")

# If the file is uploaded, load it
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Here is a preview of the uploaded data:")
    st.write(data.head())

    # Proceed with your analysis using the uploaded data if needed
    X = data.drop(['Date', 'Close'], axis=1)
    Y = data['Close']

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Fit Random Forest model
    reg = RandomForestRegressor()
    reg.fit(X_train, Y_train)

    # Make predictions and evaluate the model
    pred = reg.predict(X_test)
    score = r2_score(Y_test, pred)

    # Display the model's performance
    st.subheader('Model Performance (R2 Score)')
    st.write(score)

    # Plot the predictions against actual values
    fig, ax = plt.subplots()
    ax.plot(Y_test.values, label='Actual', color='blue')
    ax.plot(pred, label='Predicted', color='red')
    ax.legend()
    st.pyplot(fig)

else:
    # Download gold futures data for prediction if no file is uploaded
    st.write("Using default data (Gold Futures) since no file was uploaded.")
    gold = yf.download('GC=F', period='10y')

    if not gold.empty:
        # Reset the index and drop missing values
        gold.reset_index(inplace=True)
        gold.dropna(inplace=True)

        # Define features and target
        X = gold.drop(['Date', 'Close'], axis=1)
        Y = gold['Close']

        # Train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Fit Random Forest model
        reg = RandomForestRegressor()
        reg.fit(X_train, Y_train)

        # Make predictions and evaluate the model
        pred = reg.predict(X_test)
        score = r2_score(Y_test, pred)

        # Show the gold dataset and R2 score
        st.subheader('Using Random Forest Regressor')
        st.write(gold)
        st.subheader('Model Performance (R2 Score)')
        st.write(score)

        # Plot the predictions against actual values
        fig, ax = plt.subplots()
        ax.plot(Y_test.values, label='Actual', color='blue')
        ax.plot(pred, label='Predicted', color='red')
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("The dataset is empty or could not be fetched.")