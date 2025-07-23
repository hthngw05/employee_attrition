import streamlit as st
import pandas as pd
import shap 
import matplotlib.pyplot as plt
import joblib
import base64

def set_background(bg_path):
    with open(bg_path, "rb") as f:
        data_url = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: 
                linear-gradient(rgba(0, 0, 0, 0.1), rgba(0, 0, 0, 0.1)),
                url("data:image/jpeg;base64,{data_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Set the background image with an overlay
set_background("background.jpg")

#load the model and data
model = joblib.load('model.pkl')
hr_data = pd.read_csv("HR_cleaned.csv")
columns = joblib.load('columns.pkl')

# Drop target column
X = hr_data.drop(columns=['left'])

# match the model training columns
X = pd.get_dummies(X, columns=['department', 'salary'], drop_first=True)
X = X.reindex(columns=columns, fill_value=0)


# set the configuration of the app
st.set_page_config(page_title="Employee Attrition Prediction", layout="centered", page_icon=":male-detective:")
st.image("header_img.png", use_container_width=True)

# set the title of the app
st.write("""
         # 👋 Welcome to the Employee Attrition Predictor

         Wondering who might leave next? Use this app to predict **employee attrition** before it's too late!
         """)
st.write('---')
# sidebar for selecting the mode of prediction
mode = st.sidebar.radio("Select Mode", ["🧍 Individual Prediction", "📤 Batch Prediction"])


#side bar

if mode == "🧍 Individual Prediction":
    
    # header of specifying user input features
    st.sidebar.header("Specify User Input Features")

    def user_input_features():
        with st.sidebar:
            satisfaction_level = st.sidebar.slider(
                'Satisfaction Level', X['satisfaction_level'].min(), X['satisfaction_level'].max(), X['satisfaction_level'].mean(),
                help="How satisfied is the employee? (0 = not satisfied, 1 = fully satisfied)"
            )
            last_evaluation = st.sidebar.slider(
                'Last Evaluation', X['last_evaluation'].min(), X['last_evaluation'].max(), X['last_evaluation'].mean(),
                help="How well the employee was evaluated (0 to 1)"
            )
            number_project = st.sidebar.selectbox(
                'Number of Projects', 
                options=list(range(int(X['number_project'].min()) - 2, int(X['number_project'].max()) + 5)),
                index=list(range(int(X['number_project'].min()), int(X['number_project'].max()) + 1)).index(int(X['number_project'].mean()))
            )
            average_monthly_hours = st.sidebar.slider(
                'Average Monthly Hours', int(X['average_monthly_hours'].min()), int(X['average_monthly_hours'].max()), int(X['average_monthly_hours'].mean())
            )
            time_spend_company = st.sidebar.slider(
                'Time Spent at Company', int(X['time_spend_company'].min()), int(X['time_spend_company'].max()), int(X['time_spend_company'].mean()),
                help="Number of years the employee has stayed at the company"
            )
            work_accident = st.sidebar.selectbox('Work Accident', ['No', 'Yes'])
            work_accident = 1 if work_accident == 'Yes' else 0
            promotion_last_5years = st.sidebar.selectbox('Promotion in Last 5 Years', ['No', 'Yes'])
            promotion_last_5years = 1 if promotion_last_5years == 'Yes' else 0
            department_options = {
            'Sales': 'sales',
            'Accounting': 'accounting',
            'HR': 'hr',
            'Technical': 'technical',
            'Support': 'support',
            'Management': 'management',
            'IT': 'IT',
            'Product Management': 'product_mng',
            'Marketing': 'marketing',
            'Research & Development': 'RandD'
        }
            department_display = st.sidebar.selectbox('Department', list(department_options.keys()))
            department = department_options[department_display]
            salary_options = {'Low': 'low', 'Medium': 'medium', 'High': 'high'}
            salary_display = st.sidebar.selectbox('Salary Level', list(salary_options.keys()))
            salary = salary_options[salary_display]

            data = {
                'satisfaction_level': satisfaction_level,
                'last_evaluation': last_evaluation,
                'number_project': number_project,
                'average_monthly_hours': average_monthly_hours,
                'time_spend_company': time_spend_company,
                'work_accident': work_accident,
                'promotion_last_5years': promotion_last_5years,
                'department': department,
                'salary': salary
            }
            features = pd.DataFrame(data, index=[0])
            return features

    # get the user input features
    df = user_input_features()

    # display the user input features
    st.subheader('User Input Features')
    st.write(df)
    st.write('---')
    # combine the user input features with the columns
    df = pd.get_dummies(df, columns=['department', 'salary'], drop_first=True)
    # align the columns with the model
    df = df.reindex(columns=columns, fill_value=0)

    # make the prediction
    prediction = model.predict(df)

    # display the prediction probability
    st.subheader('Prediction Probability')
    prediction_proba = model.predict_proba(df)[0]
    risk_score = prediction_proba[1]
    # risk level feedback
    st.subheader('Risk Level Feedback')
    if risk_score >= 0.75:
        st.error("🔴 High risk: Strong chance the employee may leave.")
    elif risk_score >= 0.4:
        st.warning("🟡 Medium risk: Monitor engagement and workload.")
    else:
        st.success("🟢 Low risk: The employee is likely to stay.")
    st.metric("🟢 Stay Probability", f"{prediction_proba[0]*100:.1f}%")
    st.metric("🔴 Leave Probability", f"{prediction_proba[1]*100:.1f}%")

    # display the prediction result
    if prediction[0] == 1:
        st.write("The employee is likely to **leave** the company.")
    else:
        st.write("The employee is likely to **stay** with the company.")
    st.write('---')

    # explain the prediction using SHAP
    @st.cache_data
    def get_shap_values():
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return shap_values


    st.subheader('Prediction Explanation')
    st.write("Here's how each feature contributed to this prediction:")

    tab1, tab2 = st.tabs(["📊 SHAP Summary Plot", "📈 SHAP Bar Plot"])

    with st.spinner('Calculating SHAP values...'):
        shap_values = get_shap_values()
        with tab1:
            # summary plot of SHAP values
            fig1, ax1 = plt.subplots()
            shap.summary_plot(shap_values, X, show=False)
            fig1 = plt.gcf()
            st.pyplot(fig1)
            st.markdown("""
            Each dot represents one employee. The farther a dot is from the center, the more that feature influenced the prediction.

            - **Red = high value**, **Blue = low value**
            - Dots on the **right** push the prediction toward “leaving”
            - Dots on the **left** push the prediction toward “staying”

            For example: High satisfaction (red dot on the left) suggests the employee is more likely to stay.
            """)
        with tab2:
            fig_2, ax_2 = plt.subplots()
            plt.title('Feature Importance Based on SHAP Values (Bar Plot)')
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            fig_2 = plt.gcf()
            st.pyplot(fig_2)
            st.markdown("""
            This shows which features had the biggest overall impact on predictions.

            - The longer the bar, the more important the feature is for the model.
            - For example: "Satisfaction level" is the most influential factor in predicting attrition.
            """)
elif mode == "📤 Batch Prediction":
    st.subheader('Batch Prediction')
    # allow the user to upload a CSV file for batch prediction
    uploaded_file = st.file_uploader("Upload a CSV file with employee data", type="csv")

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)

        st.write("**Uploaded Data Preview:**")
        st.dataframe(uploaded_df.head())

        try:
            # preprocess the uploaded data
            uploaded_df = pd.get_dummies(uploaded_df, columns=['department', 'salary'], drop_first=True)
            uploaded_df = uploaded_df.reindex(columns=columns, fill_value=0)

            # add results column
            predictions = model.predict(uploaded_df)
            probas = model.predict_proba(uploaded_df)

            # add predictions and probabilities to the DataFrame
            uploaded_df['Prediction'] = ['Leave' if pred == 1 else 'Stay' for pred in predictions]
            uploaded_df['Leave Probability'] = [f"{proba[1]*100:.1f}%" for proba in probas]
            uploaded_df['Stay Probability'] = [f"{proba[0]*100:.1f}%" for proba in probas]

            # display the results
            st.success("✅ Predictions complete!")
            st.dataframe(uploaded_df)

            # save the results to a new CSV file
            csv_download = uploaded_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions",
                data=csv_download,
                file_name='employee_predictions.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
            st.exception(e)

    st.write('---')
