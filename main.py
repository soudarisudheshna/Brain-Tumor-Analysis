import os
import json
import openai
import streamlit as st
from dotenv import load_dotenv

# AI Model Imports
import google.generativeai as gen_ai

# Brain Tumor Classification Imports
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

import streamlit.components.v1 as components

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="TumorScanAI.com",
    page_icon=":brain:",
    layout="centered",
)

# Load the Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)  # Use the API key from the .env file
chat_model = gen_ai.GenerativeModel('gemini-pro')

# Set up the brain tumor classification model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'alexnet_brain_tumor_classification.pth'

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

alexnet = models.alexnet(pretrained=False)
alexnet.classifier[6] = nn.Linear(4096, 4)  # 4 classes: glioma, meningioma, notumor, pituitary
alexnet.load_state_dict(torch.load(model_path, map_location=device))
alexnet = alexnet.to(device)
alexnet.eval()

class_names = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']

# Function to predict the class of an uploaded image
def predict_image(image, model):
    image = image_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    return predicted_class

# Function to assess emergency level and suggest treatment or vitamins
def assess_tumor_emergency(predicted_class):
    emergency_info = {
    
  "Pituitary": {
    "emergency_level": "Medium",
    "action": "Visit a doctor to discuss treatment options. Pituitary tumors can affect hormone levels and require medical attention.",
    "origin": "Arise from the pituitary gland, a small gland at the base of the brain responsible for hormone production.",
    "description": """
        The pituitary gland is a small, pea-sized gland located at the base of the brain. 
        It is responsible for producing and releasing hormones that regulate a wide range 
        of bodily functions, including growth, metabolism, and reproduction.
        
        Pituitary tumors are growths that develop in the pituitary gland. They can be benign
         (non-cancerous) or malignant (cancerous). 
        Benign pituitary tumors are more common than malignant pituitary tumors.

        Pituitary tumors can cause a variety of symptoms, depending on their size 
        and location. Common symptoms include:
        - Headaches
        - Vision problems
        - Double vision
        - Blurred vision
        - Loss of peripheral vision
        - Nausea and vomiting
        - Fatigue
        - Weight gain
        - Increased thirst
        - Frequent urination
        - Menstrual irregularities
        - Infertility
        - Erectile dysfunction
        
        Pituitary tumors are diagnosed with a combination of physical 
        examination, blood tests, and imaging tests. Blood tests can
         be used to measure the levels of hormones produced by the pituitary 
         gland. Imaging tests, such as MRI and CT scans, can be used to 
         visualize the pituitary gland and to identify any tumors.

        Treatment for pituitary tumors depends on the size, location, and
         type of tumor. Treatment options include:
        - Surgery
        - Radiation therapy
        - Medication
        - Observation

        Surgery is the primary treatment for pituitary tumors. The goal of 
        surgery is to remove the tumor while preserving the function of the
         pituitary gland. Surgery is typically performed through the nose
          or through a small incision in the forehead.

        Radiation Therapy: Radiation therapy uses high-energy radiation to
         kill tumor cells. It can be used before surgery to shrink the tumor
          or after surgery to kill any remaining tumor cells.

        Medication: Medication can be used to treat pituitary tumors that
         are not amenable to surgery or radiation therapy. Medications can
          be used to lower hormone levels, shrink the tumor, or relieve symptoms.

        Observation: Observation is an option for patients with small,
         slow-growing pituitary tumors that are not causing any symptoms.
          Patients who are observed will have regular blood tests and imaging 
          tests to monitor the tumor.

        Prognosis: The prognosis for pituitary tumors depends on the size, location,
         and type of tumor, as well as the patient's age and overall health.
          The five-year survival rate for patients with pituitary tumors is about 95%.
    """
  },
  "Glioma": {
    "emergency_level": "High",
    "action": "Seek immediate medical consultation and treatment options, as gliomas are often aggressive and require prompt attention.",
    "origin": "Gliomas arise from glial cells, which support and protect neurons in the brain and spinal cord.",
    "description": """
        Gliomas are tumors that arise from the glial cells in the brain or spinal cord. They are the most common type of brain tumor, accounting for about 80% of all brain tumor cases. Gliomas can be classified into different grades based on their aggressiveness, ranging from grade I (least aggressive) to grade IV (most aggressive).

        Symptoms of gliomas can vary depending on the location and size of the tumor.
         Common symptoms include:
        - Headaches
        - Seizures
        - Nausea and vomiting
        - Vision or hearing problems
        - Weakness or numbness
        - Memory problems
        - Personality changes

        Gliomas are diagnosed through physical examinations, imaging tests like MRI 
        or CT scans, and biopsy to confirm the diagnosis and determine the grade of
         the tumor.

        Treatment for gliomas includes:
        - Surgery: Removing as much of the tumor as possible while preserving 
        healthy tissue.
        - Radiation therapy: High-energy radiation to kill tumor cells.
        - Chemotherapy: Drugs to kill tumor cells or stop them from growing.

        Prognosis varies depending on the tumor's grade. Lower-grade gliomas 
        have a better prognosis with treatment, but higher-grade gliomas,
        particularly grade IV gliomas (glioblastomas), have a poor 
        prognosis with a survival rate lower than 5% for five years.

        Gliomas require close monitoring and aggressive treatment, especially
        for high-grade types, to manage symptoms and improve the patient's 
        quality of life.
    """
  },
  "Meningioma": {
    "emergency_level": "Low",
    "action": "Consult a doctor for routine follow-ups and management options, as most meningiomas are benign and treatable.",
    "origin": "Meningiomas arise from the meninges, the membranes that cover the brain and spinal cord.",
    "description": """
        Meningiomas are typically benign tumors that form in the meninges, which are the protective layers surrounding the brain and spinal cord. They are the most common type of primary brain tumor, accounting for about 30% of all brain tumors. Meningiomas can grow slowly and often don't cause symptoms right away.

        Symptoms of meningiomas can vary based on their size and location.
         Common symptoms include:
        - Headaches
        - Seizures
        - Vision problems
        - Hearing issues
        - Weakness or numbness
        - Balance problems

        Diagnosis of meningiomas is usually through imaging tests like MRI 
        and CT scans, followed by biopsy to confirm the nature of the tumor.

        Treatment options for meningiomas include:
        - Surgery: Removing the tumor, especially if it is causing symptoms.
        - Radiation therapy: Used for inoperable or recurrent meningiomas to 
          shrink the tumor.
        - Observation: If the tumor is small and not causing significant
         symptoms, doctors may opt for regular monitoring rather than immediate
          intervention.

        Most meningiomas are benign and can be successfully treated with surgery.
         The prognosis is excellent for patients who undergo surgical treatment,
          with many achieving full recovery. However, for malignant meningiomas,
           the prognosis depends on the extent of tumor removal and other factors
            like age and overall health.
    """
  },
  "No Tumor": {
    "emergency_level": "None",
    "action": "No medical action required. Regular check-ups are recommended for maintaining health and well-being.",
    "origin": "No tumor detected in the body or brain. Healthy tissue.",
    "description": """
        No tumor means that there are no abnormal growths or masses present
        in the body or brain. This is the ideal scenario for an individual’s 
        health, indicating that there are no cancerous or benign growths that
         could cause harm or disrupt normal bodily functions.

        In this scenario, individuals typically experience no symptoms related
         to tumors, such as headaches, seizures, vision issues, or pain. This 
         is the result of healthy and normal functioning tissues.

        Regular health check-ups, including physical exams and imaging tests 
        (such as MRIs or CT scans when necessary), are always recommended to 
        ensure continued good health and early detection of any potential health
         concerns. Early detection of abnormal growths or tumors can be critical
          for successful treatment.

        For individuals with no tumors, the overall prognosis is excellent, as no
         medical conditions related to tumors are present.
    """
  }
    }
    
    # Check if the predicted class is in the dictionary
    if predicted_class in emergency_info:
        return emergency_info[predicted_class]
    else:
        return {
            "emergency_level": "None",
            "action": "No medical action required. Regular check-ups are recommended for maintaining health and well-being.",
            "origin": "No tumor detected in the body or brain. Healthy tissue.",
            "description": """
        No tumor means that there are no abnormal growths or masses present
        in the body or brain. This is the ideal scenario for an individual’s 
        health, indicating that there are no cancerous or benign growths that
         could cause harm or disrupt normal bodily functions.

        In this scenario, individuals typically experience no symptoms related
         to tumors, such as headaches, seizures, vision issues, or pain. This 
         is the result of healthy and normal functioning tissues.

        Regular health check-ups, including physical exams and imaging tests 
        (such as MRIs or CT scans when necessary), are always recommended to 
        ensure continued good health and early detection of any potential health
         concerns. Early detection of abnormal growths or tumors can be critical
          for successful treatment.

        For individuals with no tumors, the overall prognosis is excellent, as no
         medical conditions related to tumors are present.
    """
        }

if "page" not in st.session_state:
    st.session_state.page = "main"  # Default page

def navigate_to(page):
    st.session_state.page = page


#main page
import streamlit as st

def main_page():
    # Set the page background and text color
    st.markdown("""
        <style>
        body {
            background-color: white;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>MEDICAL AI PLATFORM</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Advanced brain tumor classification and medical chatbot assistance powered by artificial intelligence.</h3>", unsafe_allow_html=True)

    # Button and box styling
    button_style = """
        <style>
        .stButton>button {
            width: 100%;
            height: 80px;
            font-size: 24px;
            border: 2px solid black;
            background-color: white;
            color: black;
        }
        .stButton>button:hover {
            background-color: black;
            color: white;
        }
        
        /* Styling for boxes */
        .stTextInput, .stNumberInput, .stTextArea, .stSelectbox, .stCheckbox {
            border: 2px solid black;
            background-color: white;
            color: black;
        }
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Sign-In"):
            navigate_to("signin")

    with col2:
        if st.button("Register"):
            navigate_to("register")


import streamlit as st
import json
import os

# Register Page
def register_page():
    st.title("Register")

    # Input fields for username, password, and confirm password
    username = st.text_input("Enter a username:")
    password = st.text_input("Enter a password:", type="password")
    confirm_password = st.text_input("Confirm your password:", type="password")

    # Create two columns for the buttons
    col1, col2 = st.columns([1, 1])

    # Register button
    with col1:
        if st.button("Register"):
            # Validate inputs
            if username == "" or password == "" or confirm_password == "":
                st.error("All fields are required. Please fill in all the fields.")
            elif password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            else:
                # Save user credentials to JSON file
                if os.path.exists("users.json"):
                    with open("users.json", "r") as f:
                        users = json.load(f)
                else:
                    users = {}

                if username in users:
                    st.error("Username already exists. Please choose a different username.")
                else:
                    users[username] = password
                    with open("users.json", "w") as f:
                        json.dump(users, f)
                    st.success("Registration successful! You can now sign in.")
                    navigate_to("signin")

    # Sign In button (right-aligned)
    with col2:
        if st.button("Sign In"):
            navigate_to("signin")

# Sign In Page
def signin_page():
    st.title("Sign In")
    
    # Input fields for username and password
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    # Create two columns for the buttons
    col1, col2 = st.columns([1, 1])

    # Sign-In button (left-aligned)
    with col1:
        if st.button("Sign In"):
            # Validate inputs
            if username == "" or password == "":
                st.error("Both username and password are required.")
            else:
                if os.path.exists("users.json"):
                    with open("users.json", "r") as f:
                        users = json.load(f)

                    if username in users and users[username] == password:
                        st.success("Successfully signed in!")
                        navigate_to("dashboard")
                    else:
                        st.error("Invalid credentials. Please try again.")
                else:
                    st.error("No users registered yet. Please register first.")

    # Sign-Up button (right-aligned)
    with col2:
        if st.button("Sign Up"):
            navigate_to("register")



#dashboard
def dashboard_page():
    # Set the background color to white and text color to black
    st.markdown("""
        <style>
        body {
            background-color: white;
            color: black;
        }
        .stButton>button {
            width: 100%;
            height: 80px;
            font-size: 24px;
            border: 2px solid black;
            background-color: white;
            color: black;
        }
        .stButton>button:hover {
            background-color: black;
            color: white;
        }
        .logout-button {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 10;
            background-color: white;
            border: 2px solid black;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header and description
    st.markdown("<h1 style='text-align: center; color: white;'>MEDICAL AI PLATFORM</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>Advanced brain tumor classification and medical chatbot assistance powered by artificial intelligence.</h3>", unsafe_allow_html=True)

    # Create three columns for the main buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    # Buttons for each section
    with col1:
        with st.container():
            st.button("Symptom Analyzer", on_click=go_to_symptom_analyzer)

    with col2:
        with st.container():
            st.button("Brain Tumor Analyzer", on_click=go_to_brain_tumor_analyzer)

    with col3:
        with st.container():
            st.button("Chatbot", on_click=go_to_chatbot)



# Placeholder function to simulate navigating to the Symptom Analyzer page
def go_to_symptom_analyzer():
    st.session_state.page = "symptom_analyzer"

# Placeholder function to simulate navigating to the Brain Tumor Analyzer page
def go_to_brain_tumor_analyzer():
    st.session_state.page = "brain_tumor_analyzer"

# Function to initiate the chatbot page
def go_to_chatbot():
    st.session_state.page = "chatbot"

# Symptom Analyzer Page (Loading HTML file)
def symptom_analyzer_page():
    st.markdown("<h1 style='text-align: center;'>Symptom Analyser</h1>", unsafe_allow_html=True)

    # Load and display the HTML file using components
    html_file_path = 'index.html'  # Make sure the path is correct
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except UnicodeDecodeError:
        st.error("Error reading the HTML file. Please check the file encoding.")
        return

    # Display HTML content in Streamlit
    components.html(html_content, height=600, scrolling=True)

    # Back to main page
    if st.button("Go Back"):
        st.session_state.page = "dashboard"


# Brain Tumor Analyzer Page
def brain_tumor_analyzer_page():
    st.title("Brain Tumor Analyzer")
    st.markdown("Upload an MRI image to analyze a potential brain tumor.")
    uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        resized_image = image.resize((300, 300))
        st.image(resized_image, caption="Uploaded MRI Image", use_column_width=True)

        if st.button("Classify Tumor"):
            predicted_class = predict_image(image, alexnet)
            classification_result = f"The predicted class is: {predicted_class}"

            # Assess emergency level and suggest treatment
            assessment = assess_tumor_emergency(predicted_class)
            st.markdown(f"Emergency Level: {assessment['emergency_level']}")
            st.markdown(f"Recommended Action: {assessment['action']}")
            st.markdown(f"Origin: {assessment['origin']}")
            st.markdown(f"Description: {assessment['description']}")

    # Back to main page
    if st.button("Go Back"):
        st.session_state.page = "dashboard"


# Function to translate chatbot role into Streamlit's format
def translate_role_for_streamlit(role):
    if role == "user":
        return "user"
    elif role == "assistant":
        return "assistant"
    else:
        return "unknown"

# Chatbot Page (Modified for ChatGPT-like behavior)
def chatbot_section():
    st.header("Chat with Our Brainy Bot")

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = chat_model.start_chat(history=[])

    # Location Input Section
    st.subheader("Find Nearby Hospitals")

    # Input field for user's location
    user_location = st.text_input("Enter your area or city to find nearby hospitals:", key="location_input")
    if user_location:
        # Store the location in session state
        st.session_state.user_location = user_location

        # Send location to chatbot
        st.session_state.chat_session.send_message(f"User is looking for hospitals in: {user_location}")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Type your message here...")
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro (or a similar AI) and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

    # Back to main page (this is now at the end)
    if st.button("Go Back"):
        st.session_state.page = "dashboard"





# Initialize page state
# App logic for navigation
if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "signin":
    signin_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
elif st.session_state.page == "symptom_analyzer":
    symptom_analyzer_page()
elif st.session_state.page == "brain_tumor_analyzer":
    brain_tumor_analyzer_page()
elif st.session_state.page == "chatbot":
    chatbot_section()