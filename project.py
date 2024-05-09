import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import gdown
import streamlit.components.v1 as components
import os

# Function to load and preprocess image
def preprocess_image(image):
    # Convert image to RGB if it has less than 3 channels
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image to the model input size
    image = image.resize((256, 256))
    
    # Convert the image to a numpy array and normalize pixel values
    processed_image = np.array(image) / 255.0
    
    # Ensure image has three channels (RGB)
    if processed_image.shape[-1] != 3:
        processed_image = np.stack((processed_image,) * 3, axis=-1)
    
    return processed_image

# Function to make glaucoma prediction
def predict_glaucoma(image, classifier):
    image = np.expand_dims(image, axis=0)
    prediction = classifier.predict(image)
    if prediction[0][0] > prediction[0][1]:
        return "Glaucoma"
    else:
        return "Normal"

# Function to clear old results
def clear_results():
    if os.path.exists("results.csv"):
        os.remove("results.csv")

# Google Drive file ID
file_id = '1lhBtxhP18L-KA7wDh4N72xTHZMLUZT82'

# Define the destination path for the model file
model_path = 'combinee_cnn.h5'

# Download the model file from Google Drive
if not os.path.exists(model_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        gdown.download(url, model_path, quiet=False)
    except Exception as e:
        st.error(f"Error downloading the model: {e}")

# Load pretrained model
try:
    classifier = load_model(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Define the background image URL
background_image_url = "https://img.freepik.com/free-photo/security-access-technologythe-scanner-decodes-retinal-data_587448-5015.jpg"

# Set background image using HTML
background_image_style = f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        height: 100vh;  /* Adjust the height as needed */
        width: 100vw;   /* Adjust the width as needed */
    }}
    .blue-bg {{
        background-color: darkblue; /* Changed to dark blue */
        padding: 20px;  /* Increased padding */
        margin-bottom: 20px; /* Added margin */
        color: white;   /* Text color */
    }}
    .white-bg {{
        background-color: white; /* Changed to white */
    }}
    .bold-text {{
        font-weight: bold; /* Added bold font weight */
        font-size: larger; /* Increased font size */
        padding: 5px; /* Adjusted padding */
        color: black; /* Text color */
    }}
    .red-bg {{
        background-color: red;
        padding: 10px;  /* Adjust the padding as needed */
        margin: 10px;   /* Adjust the margin as needed */
        color: white;   /* Text color */
    }}
    .green-bg {{
        background-color: green;
        padding: 10px;  /* Adjust the padding as needed */
        margin: 10px;   /* Adjust the margin as needed */
        color: white;   /* Text color */
    }}
    .yellow-bg {{
        background-color: yellow;
        padding: 10px;  /* Adjust the padding as needed */
        margin: 10px;   /* Adjust the margin as needed */
        color: black;   /* Text color */
    }}
    </style>
"""

# Display background image using HTML
st.markdown(background_image_style, unsafe_allow_html=True)

# Set title with transition effects and font styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #ecf0f1;
            padding: 20px;
            font-family: 'Montserrat', sans-serif;
            
        }

        @keyframes fadein {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
    </style>
    <h1 class='title blue-bg'>GlaucoGuard: Gaining Clarity in Glaucoma diagnosis through Deep Learning</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# Paragraph with content about uploading fundus images
st.markdown("""<p style='font-size: 20px; text-align: center; background-color: orange; color: black;'>This is a simple image classification web application to predict glaucoma through fundus images of the eye. <strong><em>Please upload fundus images only.</em></strong></p>""", unsafe_allow_html=True)

st.markdown("---")

# Initialize DataFrame for results
if not os.path.exists("results.csv"):
    all_results = pd.DataFrame(columns=["Image", "Prediction"])
else:
    all_results = pd.read_csv("results.csv")

# Sidebar for uploading image
st.markdown("""<p style='font-size: 20px;  background-color: cyan; color: black;'>Upload an image for glaucoma detection (Max size: 200 MB)</p>""", unsafe_allow_html=True)
st.empty()
uploaded_file = st.file_uploader(" ",type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="file_uploader", help="Upload an image for glaucoma detection (Max size: 200 MB)")
st.markdown("""
    <style>
        div[data-testid="stBlock"] {
            background-color: white;
            padding: 10px; /* Adjust padding as needed */
            border-radius: 10px; /* Add rounded corners */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Add shadow for depth */
        }
    </style>
""", unsafe_allow_html=True)

if st.button("Clear Results"):
    all_results = pd.DataFrame(columns=["Image", "Prediction"])
    all_results.to_csv("results.csv", index=False)
    st.success("Results cleared successfully!")
    st.stop()
# Main content area
if uploaded_file is not None:

    # Display uploaded image
    try:
        original_image = Image.open(uploaded_file)
        st.image(original_image,  use_column_width=True)
        st.markdown("<div style='background-color: white; text-align: center; padding: 5px'><strong>Uploaded Image</strong></div>", unsafe_allow_html=True)
        
        # Perform glaucoma detection
        with st.spinner("Detecting glaucoma..."):
            processed_image = preprocess_image(original_image)
            prediction = predict_glaucoma(processed_image, classifier)

        # Customize messages based on prediction
        if prediction == "Glaucoma":
            st.markdown("<p class='red-bg'>Your eye is diagnosed with Glaucoma. Please consult an ophthalmologist.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='green-bg'>Your eyes are healthy.</p>", unsafe_allow_html=True)

        # Add new result to DataFrame
        new_result = pd.DataFrame({"Image": [uploaded_file.name], "Prediction": [prediction]})
        all_results = pd.concat([new_result, all_results], ignore_index=True)
        
        if not all_results.empty:
            st.markdown("<h3  class='blue-bg' style='color: white;'>Detection Results</h3>", unsafe_allow_html=True)
            st.dataframe(all_results.style.applymap(lambda x: 'color: red' if x == 'Glaucoma' else 'color: green', subset=['Prediction']))

        # Save updated results to CSV
        all_results.to_csv("results.csv", index=False)

        # Pie chart
        st.markdown("<h3  style='color: white; background-color: darkblue'>Pie Chart</h3>", unsafe_allow_html=True)
        pie_data = all_results['Prediction'].value_counts()
        fig, ax = plt.subplots()
        colors = ['green' if label == 'Normal' else 'red' for label in pie_data.index]
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

        # Bar chart
        st.markdown("<h3   style='color: white; background-color: darkblue'>Bar Chart</h3>", unsafe_allow_html=True)
        bar_data = all_results['Prediction'].value_counts()
        fig, ax = plt.subplots()
        colors = ['green' if label == 'Normal' else 'red' for label in bar_data.index]
        ax.bar(bar_data.index, bar_data, color=colors)
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        # Option to download prediction report
        st.markdown("<h3  class='blue-bg' style='color: white;'>Download Prediction Report</h3>", unsafe_allow_html=True)
        csv = all_results.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="prediction_report.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
else:
    st.markdown("<p style='font-size: 20px;  background-color: cyan; color: black;'>No images uploaded yet.</p>", unsafe_allow_html=True)

st.markdown("""
    <div class="white-bg" style="padding: 20px; text-align: center;">
        <h2>Thank you for using GlaucoGuard!</h2>
        <p>This application was developed by Chaithanya S , Chinmayi H , Elaine M Paily & Hamsini D .</p>
        <p>Department of CSE, Vidyavardhaka College of Engineering Mysore</p>
    </div>
""", unsafe_allow_html=True)
