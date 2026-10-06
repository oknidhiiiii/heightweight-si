import streamlit as st
import pickle
import pandas as pd

st.set_page_config(
    page_title="Height Weight Predictor",
    page_icon="📊",
    layout="wide"
)

with open("height_weight_model.pkl", "rb") as file:
    model = pickle.load(file)

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Height & Weight Predictor")
st.write("Machine Learning based height-to-weight prediction")

st.divider()

st.sidebar.header("Enter Height")

height = st.sidebar.number_input(
    "Height (Inches)",
    min_value=40.0,
    max_value=90.0,
    value=65.0,
    step=0.5
)

predict_button = st.sidebar.button(
    "Predict Weight",
    type="primary"
)

reset_button = st.sidebar.button("Reset History")

if reset_button:
    st.session_state.history = []
    st.rerun()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Height", f"{height:.1f} in")

with col2:
    st.metric("Height", f"{height * 2.54:.1f} cm")

with col3:
    st.metric("Height", f"{height / 12:.2f} ft")

st.divider()

if predict_button:

    input_data = pd.DataFrame({
        "Height(Inches)": [height]
    })

    prediction = model.predict(input_data)

    weight = float(prediction[0])

    height_m = height * 0.0254
    weight_kg = weight * 0.453592

    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    st.session_state.history.append({
        "Height (inches)": round(height, 1),
        "Predicted Weight (lbs)": round(weight, 2),
        "BMI": round(bmi, 2),
        "Category": category
    })

    st.subheader("Prediction Result")

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "Predicted Weight",
            f"{weight:.2f} lbs"
        )

    with result2:
        st.metric(
            "BMI",
            f"{bmi:.2f}"
        )

    with result3:
        st.metric(
            "Category",
            category
        )

    st.success(
        f"For a height of {height:.1f} inches, "
        f"the predicted weight is {weight:.2f} pounds."
    )

    st.subheader("Prediction Visualization")

    chart_data = pd.DataFrame({
        "Measurement": ["Height", "Predicted Weight"],
        "Value": [height, weight]
    })

    st.bar_chart(
        chart_data.set_index("Measurement")
    )

if st.session_state.history:

    st.divider()

    st.subheader("Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    st.subheader("Height vs Predicted Weight")

    st.line_chart(
        history_df.set_index("Height (inches)")[
            "Predicted Weight (lbs)"
        ]
    )

st.divider()

with st.expander("About This ML Model"):

    st.write(
        "This application uses the trained model "
        "stored in height_weight_model.pkl."
    )

    st.write("Input: Height (Inches)")
    st.write("Output: Predicted Weight (Pounds)")

st.caption("Height & Weight Prediction | Machine Learning Application")