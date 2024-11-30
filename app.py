import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle

#####################################
# Set Streamlit to full screen mode##
#####################################
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🌸",
    layout="wide",  # Adjust layout as needed
    initial_sidebar_state="expanded",
)

#####################################
# Load and Predict Function
#####################################

def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model


def main():
    st.title('Diabetes Prediction')
    selected = option_menu(
        menu_title=None,
        options=["Home", "App", "Contact"],
        icons=["house", "app-indicator", "envelope"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "25px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "green"},
        },
    )
###################################################################
    if selected == "Home":
        st.image("MoVies.png")
######################################################################
    elif selected == "App":
        # Load the saved models
        logistic_model = load_model('logit_model.pkl')
        svm_model = load_model('svc_model.pkl')
        k_nearest_model = load_model('k_nearest_model.pkl')
        dt_model = load_model('dt_model.pkl')
        rf_model = load_model('rf_model.pkl')
        xgb_model = load_model('xgb_model.pkl')
        clf_model = load_model('clf_model.pkl')

        st.title('Health Diagnosis Prediction App')

        st.write("""
        This app predicts the **health diagnosis** based on user inputs.
        """)

        # Model selection
        model_choice = st.sidebar.selectbox('Select Model', options=[
            'Logistic Regression', 'SVM', 'K-Nearest Neighbors', 
            'Decision Tree', 'Random Forest', 'Gradient Boosting Classifier', 
            'XGBoost','Stacking'
        ])

        # Create sliders and selectboxes for user input
        age = st.slider('Age', 0, 80, 50)
        gender = st.selectbox('Gender', options=['Male', 'Female'])
        gender = 0 if gender == 'Male' else 1
        bmi = st.slider('BMI', 10, 50, 25)
        high_bp = st.selectbox('High Blood Pressure', options=['No', 'Yes'])
        high_bp = 0 if high_bp == 'No' else 1
        fbs = st.slider('Fasting Blood Sugar (mg/dL)', 70, 200, 100)
        hba1c_level = st.slider('HbA1c Level', 3.0, 15.0, 5.0)
        smoking = st.selectbox('Smoking', options=['No', 'Yes'])
        smoking = 0 if smoking == 'No' else 1

        # Prepare input data
        input_data = np.array([[age, gender, bmi, high_bp, fbs, hba1c_level, smoking]])

        # Button for prediction
        if st.button('Predict'):
            if model_choice == 'Logistic Regression':
                model = logistic_model
            elif model_choice == 'SVM':
                model = svm_model
            elif model_choice == 'K-Nearest Neighbors':
                model = k_nearest_model
            elif model_choice == 'Decision Tree':
                model = dt_model
            elif model_choice == 'Random Forest':
                model = rf_model
            elif model_choice == 'Stacking':
                model = clf_model
            else:
                model = xgb_model

            prediction = model.predict(input_data)[0]

            st.subheader('Prediction')
            st.write(f'The predicted diagnosis is **{"Healthy" if prediction == 0 else "Unhealthy"}**.')


####################################################################################################
  

#########################################################
    elif selected == "Contact":
    ### About the author
        st.write("##### About the author:")
        
        ### Author name
        st.write("<p style='color:blue; font-size: 50px; font-weight: bold;'>Usama Munawar</p>", unsafe_allow_html=True)
        
        ### Connect on social media
        st.write("##### Connect with me on social media")
        
        ### Add social media links
        ### URLs for images
        linkedin_url = "https://img.icons8.com/color/48/000000/linkedin.png"
        github_url = "https://img.icons8.com/fluent/48/000000/github.png"
        youtube_url = "https://img.icons8.com/?size=50&id=19318&format=png"
        twitter_url = "https://img.icons8.com/color/48/000000/twitter.png"
        facebook_url = "https://img.icons8.com/color/48/000000/facebook-new.png"
        
        ### Redirect URLs
        linkedin_redirect_url = "https://www.linkedin.com/in/abu--usama"
        github_redirect_url = "https://github.com/UsamaMunawarr"
        youtube_redirect_url ="https://www.youtube.com/@CodeBaseStats"
        twitter_redirect_url = "https://twitter.com/Usama__Munawar?t=Wk-zJ88ybkEhYJpWMbMheg&s=09"
        facebook_redirect_url = "https://www.facebook.com/profile.php?id=100005320726463&mibextid=9R9pXO"
        
        ### Add links to images
        st.markdown(f'<a href="{github_redirect_url}"><img src="{github_url}" width="60" height="60"></a>'
                    f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}" width="60" height="60"></a>'
                    f'<a href="{youtube_redirect_url}"><img src="{youtube_url}" width="60" height="60"></a>'
                    f'<a href="{twitter_redirect_url}"><img src="{twitter_url}" width="60" height="60"></a>'
                    f'<a href="{facebook_redirect_url}"><img src="{facebook_url}" width="60" height="60"></a>', unsafe_allow_html=True)
    # Thank you message
    st.write("<p style='color:green; font-size: 30px; font-weight: bold;'>Thank you for using this app, share with your friends!😇</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

