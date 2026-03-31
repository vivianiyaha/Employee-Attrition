import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('employee_attrition.csv')

# Convert Attrition to numeric (Yes=1, No=0)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Sidebar
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Predictor', 'About', 'Profile'])

# Predictor Page
if page == 'Predictor':
    st.title('Employee Attrition Predictor')

    # Select features
    X = df[['Age', 'DailyRate', 'DistanceFromHome', 'Education',
            'EnvironmentSatisfaction', 'HourlyRate', 'JobLevel',
            'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
            'PercentSalaryHike', 'PerformanceRating',
            'RelationshipSatisfaction', 'StockOptionLevel',
            'TotalWorkingYears', 'TrainingTimeLastYear',
            'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion',
            'YearsWithCurrManager']]

    y = df['Attrition']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # User inputs
    st.header('Enter Employee Information:')

    age = st.number_input('Age', 18, 60, 30)
    daily_rate = st.number_input('Daily Rate', 100, 1500, 500)
    distance = st.number_input('Distance From Home', 1, 50, 10)
    education = st.selectbox('Education (1-5)', [1, 2, 3, 4, 5])
    env_sat = st.selectbox('Environment Satisfaction (1-4)', [1, 2, 3, 4])
    hourly_rate = st.number_input('Hourly Rate', 10, 100, 50)
    job_level = st.selectbox('Job Level (1-5)', [1, 2, 3, 4, 5])
    job_sat = st.selectbox('Job Satisfaction (1-4)', [1, 2, 3, 4])
    income = st.number_input('Monthly Income', 1000, 20000, 5000)
    companies = st.number_input('Num Companies Worked', 0, 10, 2)
    hike = st.number_input('Percent Salary Hike', 10, 30, 15)
    performance = st.selectbox('Performance Rating (1-4)', [1, 2, 3, 4])
    relationship = st.selectbox('Relationship Satisfaction (1-4)', [1, 2, 3, 4])
    stock = st.selectbox('Stock Option Level (0-3)', [0, 1, 2, 3])
    total_years = st.number_input('Total Working Years', 0, 40, 5)
    training = st.number_input('Training Time Last Year', 0, 10, 3)
    worklife = st.selectbox('Work Life Balance (1-4)', [1, 2, 3, 4])
    years_company = st.number_input('Years at Company', 0, 40, 5)
    years_role = st.number_input('Years in Current Role', 0, 20, 3)
    years_promo = st.number_input('Years Since Last Promotion', 0, 15, 1)
    years_manager = st.number_input('Years With Current Manager', 0, 20, 3)

    # Input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'DailyRate': [daily_rate],
        'DistanceFromHome': [distance],
        'Education': [education],
        'EnvironmentSatisfaction': [env_sat],
        'HourlyRate': [hourly_rate],
        'JobLevel': [job_level],
        'JobSatisfaction': [job_sat],
        'MonthlyIncome': [income],
        'NumCompaniesWorked': [companies],
        'PercentSalaryHike': [hike],
        'PerformanceRating': [performance],
        'RelationshipSatisfaction': [relationship],
        'StockOptionLevel': [stock],
        'TotalWorkingYears': [total_years],
        'TrainingTimeLastYear': [training],
        'WorkLifeBalance': [worklife],
        'YearsAtCompany': [years_company],
        'YearsInCurrentRole': [years_role],
        'YearsSinceLastPromotion': [years_promo],
        'YearsWithCurrManager': [years_manager]
    })

    # Prediction
    if st.button('Predict Attrition'):
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error('This employee is likely to leave the company.')
        else:
            st.success('This employee is likely to stay.')

# About Page
elif page == 'About':
    st.title('About')
    st.write('This app predicts employee attrition using machine learning.')
    st.write('It helps HR teams identify employees who may leave and take proactive actions.')

# Profile Page
elif page == 'Profile':
    st.title('Profile')
    st.write('Vivian Iyaha is a Management graduate with interest in HR analytics, Machine Learning, and AI.')
