import streamlit as st 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv('employee-attrition.csv')


# Encode categorical variables
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

df['BusinessTravel'] = df['BusinessTravel'].map({
    'Travel_Rarely': 0,
    'Travel_Frequently': 1,
    'Non-Travel': 2
})

df['Department'] = df['Department'].map({
    'Sales': 0,
    'Research & Development': 1,
    'Human Resources': 2
})

df['MaritalStatus'] = df['MaritalStatus'].map({
    'Single': 0,
    'Married': 1,
    'Divorced': 2
})

df['EducationField'] = df['EducationField'].map({
    'Life Sciences': 0,
    'Medical': 1,
    'Marketing': 2,
    'Technical Degree': 3,
    'Human Resources': 4,
    'Other': 5
})

df['JobRole'] = df['JobRole'].map({
    'Sales Executive': 0,
    'Research Scientist': 1,
    'Laboratory Technician': 2,
    'Manufacturing Director': 3,
    'Healthcare Representative': 4,
    'Manager': 5,
    'Sales Representative': 6,
    'Research Director': 7,
    'Human Resources': 8
})


# Drop unnecessary columns
df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)


# Sidebar
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Predictor', 'About', 'Profile'])


# ============================
# Predictor Page
# ============================
if page == 'Predictor':

    st.title('Employee Attrition Predictor')

    # Features & Target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Prediction function
    def predict_attrition(data):
        prediction = clf.predict(data)
        return prediction

    st.header('Enter Employee Details:')

    # Inputs
    age = st.number_input('Age', 18, 60, 30)
    daily_rate = st.number_input('Daily Rate', 100, 1500, 800)
    distance = st.number_input('Distance From Home', 1, 30, 5)
    education = st.selectbox('Education', [1, 2, 3, 4, 5])

    environment = st.selectbox('Environment Satisfaction', [1, 2, 3, 4])
    job_involvement = st.selectbox('Job Involvement', [1, 2, 3, 4])
    job_level = st.selectbox('Job Level', [1, 2, 3, 4, 5])
    job_satisfaction = st.selectbox('Job Satisfaction', [1, 2, 3, 4])

    monthly_income = st.number_input('Monthly Income', 1000, 20000, 5000)
    num_companies = st.selectbox('Num Companies Worked', list(range(10)))

    percent_hike = st.selectbox('Percent Salary Hike', list(range(11, 26)))
    performance = st.selectbox('Performance Rating', [3, 4])

    relationship = st.selectbox('Relationship Satisfaction', [1, 2, 3, 4])
    stock = st.selectbox('Stock Option Level', [0, 1, 2, 3])

    work_life = st.selectbox('Work Life Balance', [1, 2, 3, 4])

    years_total = st.number_input('Total Working Years', 0, 40, 5)
    years_company = st.number_input('Years At Company', 0, 40, 5)
    years_role = st.number_input('Years In Current Role', 0, 20, 3)
    years_promo = st.number_input('Years Since Last Promotion', 0, 15, 1)
    years_manager = st.number_input('Years With Current Manager', 0, 20, 3)

    training = st.selectbox('Training Times Last Year', list(range(7)))

    # Encoded categorical inputs
    gender = st.selectbox('Gender', ['Male', 'Female'])
    gender = 1 if gender == 'Male' else 0

    overtime = st.selectbox('Overtime', ['Yes', 'No'])
    overtime = 1 if overtime == 'Yes' else 0

    business = st.selectbox('Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
    business = {'Travel_Rarely': 0, 'Travel_Frequently': 1, 'Non-Travel': 2}[business]

    dept = st.selectbox('Department', ['Sales', 'R&D', 'HR'])
    dept = {'Sales': 0, 'R&D': 1, 'HR': 2}[dept]

    marital = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    marital = {'Single': 0, 'Married': 1, 'Divorced': 2}[marital]

    education_field = st.selectbox('Education Field',
                                  ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'HR', 'Other'])
    education_field = {
        'Life Sciences': 0, 'Medical': 1, 'Marketing': 2,
        'Technical Degree': 3, 'HR': 4, 'Other': 5
    }[education_field]

    job_role = st.selectbox('Job Role',
                            ['Sales Executive', 'Research Scientist', 'Lab Technician',
                             'Manager', 'HR'])
    job_role = {
        'Sales Executive': 0,
        'Research Scientist': 1,
        'Lab Technician': 2,
        'Manager': 5,
        'HR': 8
    }[job_role]


    # Input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'DailyRate': [daily_rate],
        'DistanceFromHome': [distance],
        'Education': [education],
        'HourlyRate': [hourly_rate],
        'MonthlyRate': [monthly_rate],
        'EnvironmentSatisfaction': [environment],
        'JobInvolvement': [job_involvement],
        'JobLevel': [job_level],
        'JobSatisfaction': [job_satisfaction],
        'MonthlyIncome': [monthly_income],
        'NumCompaniesWorked': [num_companies],
        'PercentSalaryHike': [percent_hike],
        'PerformanceRating': [performance],
        'RelationshipSatisfaction': [relationship],
        'StockOptionLevel': [stock],
        'WorkLifeBalance': [work_life],
        'TotalWorkingYears': [years_total],
        'YearsAtCompany': [years_company],
        'YearsInCurrentRole': [years_role],
        'YearsSinceLastPromotion': [years_promo],
        'YearsWithCurrManager': [years_manager],
        'TrainingTimesLastYear': [training],
        'Gender': [gender],
        'OverTime': [overtime],
        'BusinessTravel': [business],
        'Department': [dept],
        'MaritalStatus': [marital],
        'EducationField': [education_field],
        'JobRole': [job_role]
    })


    # Predict
    if st.button('Predict'):
        prediction = predict_attrition(input_data)

        if prediction[0] == 0:
            st.success('Employee is likely to stay.')
        else:
            st.error('Employee is likely to leave (Attrition).')


# ============================
# About Page
# ============================
elif page == 'About':
    st.title('About')
    st.write('This app predicts employee attrition using a Random Forest model.')
    st.write('It helps HR teams identify employees who may leave and take action early.')


# ============================
# Profile Page
# ============================
elif page == 'Profile':
    st.title('Profile')
    st.write('Vivian Iyaha is passionate about HR analytics and Machine Learning.')
