import streamlit as st
import pandas as pd
import time

st.title("Real-time CSV Visualization")

# Function to read CSV file and return DataFrame
@st.cache_data
def read_csv(file_path,columns):
    return pd.read_csv(file_path,names=columns)

# Path to your CSV file
csv_file_path = "assets/logs.csv"
table_container = st.empty()

# Main loop to periodically check for updates
while True:
    # Read the CSV file
    columns = ["timestamp","stID","CamID"]
    df = read_csv(csv_file_path,columns)
    table_container.dataframe(df)
    # Add a pause to control the update frequency (adjust as needed)
    time.sleep(5)  # Wait for 5 seconds before checking again
    # add a button to stop the loop
    

