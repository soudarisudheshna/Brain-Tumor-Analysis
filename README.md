#This project leverages deep learning to enhance brain tumor visualization by fusing MRI and CT images using the VGG Network.
It integrates advanced detection techniques with YOLOv7, enriched by attention modules and feature pyramids. An NLP-powered 
chatbot provides real-time support for healthcare professionals, answering queries about tumor types, stages, and treatments,
alongside offering personalized recommendations.

#The goal is to improve diagnosis, treatment planning, and patient management in neuro-onc

#Dataset Structure
Ensure your dataset is organized in the following structure:

plaintext
Copy code
dataset/
    train/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
    val/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
Replace class1 and class2 with your class names (e.g., tumor and normal).

#Usage
Step 1: Data Preparation
Step 2: Training the Model
step 3: Saving and Loading the Model


#Streamlit Web Application
1. Installing Streamlit
2. Web App Structure
The Streamlit app consists of different pages:

Sign-In Page: Allows users to log in.
Register Page: Lets new users register.
Dashboard: Contains Symptom Analyzer, Brain Tumor Analyzer, and Chatbot sections.

3. 5. Embedding HTML for Symptom Analyzer
If you have a pre-built HTML file for the Symptom Analyzer, you can embed it using st.components.v1.html in Streamlit.

4. Project Details JSON
You can store the user's project details, including the model, dataset path, training parameters, and best validation accuracy, in a JSON file

Acknowledgments
Streamlit: For providing an easy way to build interactive web applications.
PyTorch: For offering a powerful deep learning framework.
AlexNet: For serving as the base architecture for the classification model.

