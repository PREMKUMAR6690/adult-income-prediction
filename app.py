import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AI Income Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #080D1B,
        #111B35,
        #10172A
    );
    color: #F8FAFC;
}

/* Main content width */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0B1224;
    border-right: 1px solid #253252;
}

/* Headings */
h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
}

/* Text */
p, label, span {
    color: #CBD5E1;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: #182541 !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(23, 37, 66, 0.65);
    border: 1px solid #293B60;
    border-radius: 16px;
    padding: 15px;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(
        90deg,
        #2563EB,
        #7C3AED
    );
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
}

/* Button hover */
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border: 1px solid #93C5FD;
    color: white;
}

/* Divider */
hr {
    border-color: #293B60;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #172541;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #293B60;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# LOAD ML MODEL
# --------------------------------


MODEL_PATH = Path(__file__).parent / "models" / "adult_income_best_model.pkl"

model = joblib.load(MODEL_PATH)



# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:

    st.markdown("## 💰 IncomeAI")

    st.caption("Machine Learning Prediction System")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 About Project",
            "⚙️ How It Works"
        ]
    )

    st.divider()

    st.caption("PROJECT INFORMATION")
    st.write("Model: Adult Census Dataset")
    st.write("Task: Binary Classification")
    st.write("Framework: Streamlit")


# --------------------------------
# HOME PAGE
# --------------------------------

if page == "🏠 Home":

    st.markdown(
        "<h1 style='font-size:38px;'>"
        "💰 AI-Powered Income Prediction"
        "</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "Predict whether an individual's annual income "
        "is above or below **$50,000** using Machine Learning."
    )

    st.write("")

    # Dashboard information cards

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Prediction Type", "Classification")

    with c2:
        st.metric("Income Threshold", "$50K")

    with c3:
        st.metric("Technology", "Machine Learning")

    st.write("")

    st.markdown("## 📝 Enter Individual Details")

    st.caption(
        "Provide the following information to generate "
        "an income prediction."
    )

    # --------------------------------
    # INPUT FORM
    # --------------------------------

    with st.form("income_prediction_form"):

        st.markdown("### 👤 Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age",
                min_value=17,
                max_value=100,
                value=30
            )

        with col2:
            sex = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

        with col3:
            race = st.selectbox(
                "Race",
                [
                    "White",
                    "Black",
                    "Asian-Pac-Islander",
                    "Amer-Indian-Eskimo",
                    "Other"
                ]
            )

        col1, col2 = st.columns(2)

        with col1:
            marital_status = st.selectbox(
                "Marital Status",
                [
                    "Never-married",
                    "Married-civ-spouse",
                    "Divorced",
                    "Separated",
                    "Widowed",
                    "Married-spouse-absent",
                    "Married-AF-spouse"
                ]
            )

        with col2:
            relationship = st.selectbox(
                "Relationship",
                [
                    "Not-in-family",
                    "Husband",
                    "Wife",
                    "Own-child",
                    "Unmarried",
                    "Other-relative"
                ]
            )

        st.divider()

        st.markdown("### 💼 Employment Information")

        col1, col2 = st.columns(2)

        with col1:
            workclass = st.selectbox(
                "Work Class",
                [
                    "Private",
                    "Self-emp-not-inc",
                    "Self-emp-inc",
                    "Federal-gov",
                    "Local-gov",
                    "State-gov",
                    "Without-pay",
                    "Never-worked"
                ]
            )

            occupation = st.selectbox(
                "Occupation",
                [
                    "Adm-clerical",
                    "Exec-managerial",
                    "Handlers-cleaners",
                    "Prof-specialty",
                    "Other-service",
                    "Sales",
                    "Craft-repair",
                    "Transport-moving",
                    "Farming-fishing",
                    "Machine-op-inspct",
                    "Tech-support",
                    "Protective-serv",
                    "Armed-Forces",
                    "Priv-house-serv"
                ]
            )

        with col2:
            education = st.selectbox(
                "Education",
                [
                    "Bachelors",
                    "HS-grad",
                    "Some-college",
                    "Masters",
                    "Assoc-voc",
                    "Assoc-acdm",
                    "11th",
                    "10th",
                    "7th-8th",
                    "9th",
                    "12th",
                    "5th-6th",
                    "1st-4th",
                    "Preschool",
                    "Prof-school",
                    "Doctorate"
                ]
            )

            native_country = st.selectbox(
                "Native Country",
                [
                    "United-States",
                    "India",
                    "Mexico",
                    "Philippines",
                    "Germany",
                    "Canada",
                    "Other"
                ]
            )

        education_mapping = {
            "Preschool": 1,
            "1st-4th": 2,
            "5th-6th": 3,
            "7th-8th": 4,
            "9th": 5,
            "10th": 6,
            "11th": 7,
            "12th": 8,
            "HS-grad": 9,
            "Some-college": 10,
            "Assoc-voc": 11,
            "Assoc-acdm": 12,
            "Bachelors": 13,
            "Masters": 14,
            "Prof-school": 15,
            "Doctorate": 16
        }

        education_num = education_mapping[education]

        st.divider()

        st.markdown("### 💵 Financial Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            fnlwgt = st.number_input(
                "Final Weight",
                min_value=1,
                value=200000
            )

        with col2:
            capital_gain = st.number_input(
                "Capital Gain",
                min_value=0,
                value=0
            )

        with col3:
            capital_loss = st.number_input(
                "Capital Loss",
                min_value=0,
                value=0
            )

        hours_per_week = st.slider(
            "Working Hours Per Week",
            min_value=1,
            max_value=99,
            value=40
        )

        st.write("")

        submitted = st.form_submit_button(
            "✨ Predict Annual Income",
            use_container_width=True
        )

    # --------------------------------
    # PREDICTION
    # --------------------------------

    if submitted:

        if model is None:

            st.error(
                "Model file not found. Please check model.pkl."
            )

        else:

            input_data = pd.DataFrame([{
                "age": age,
                "workclass": workclass,
                "fnlwgt": fnlwgt,
                "education": education,
                "education_num": education_num,
                "marital_status": marital_status,
                "occupation": occupation,
                "relationship": relationship,
                "race": race,
                "sex": sex,
                "capital_gain": capital_gain,
                "capital_loss": capital_loss,
                "hours_per_week": hours_per_week,
                "native_country": native_country
            }])

            try:

                prediction = model.predict(input_data)[0]

                st.divider()

                st.markdown("## 📊 Prediction Result")

                # Assumes numeric 1 means >50K
                # and numeric 0 means <=50K.
                if str(prediction).strip() in ["1", "1.0", ">50K"]:
                    result = "Above $50K"
                    st.success("🎉 Predicted Income: Above $50K")

                elif str(prediction).strip() in ["0", "0.0", "<=50K"]:
                    result = "Below or Equal to $50K"
                    st.info("Predicted Income: $50K or Below")

                else:
                    result = str(prediction)
                    st.success(f"Predicted Income: {result}")

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(input_data)[0]

                    confidence = max(probabilities) * 100

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                st.caption(
                    "This is a machine learning estimate, "
                    "not a guarantee of actual income."
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# --------------------------------
# ABOUT PROJECT
# --------------------------------

elif page == "📊 About Project":

    st.title("📊 About This Project")

    st.write(
        "Adult Census Income Prediction is a machine learning "
        "classification project that estimates whether an "
        "individual earns more than $50,000 annually."
    )

    st.markdown("### Project Objectives")

    st.write("- Analyze individual demographic information.")
    st.write("- Study employment and financial attributes.")
    st.write("- Apply machine learning classification.")
    st.write("- Predict annual income categories.")

    st.markdown("### Dataset")

    st.write("Adult Census Income Dataset")


# --------------------------------
# HOW IT WORKS
# --------------------------------

elif page == "⚙️ How It Works":

    st.title("⚙️ How Does It Work?")

    st.markdown("""
    1. User enters personal information.

    2. The application collects employment
       and financial details.

    3. Input data is passed to the trained
       machine learning model.

    4. The model processes the information.

    5. The application displays the predicted
       income category.
    """)

    st.info(
        "The prediction depends on the trained "
        "machine learning model, Thankyou "
        
    )
   
# Professional Developer Footer

import streamlit as st

st.markdown(
    """
    <div style="text-align:center;">
        <h2 style="color:#00C896;">
            Developed by Prem Kumar
        </h2>
        <p>Machine Learning Developer</p>
        <p>Adult Census Income Prediction</p>
        <p>© 2026 Prem Kumar</p>
    </div>
    """,
    unsafe_allow_html=True
)