import pickle
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import base64

def add_bg_from_local(image_files):
    with open(image_files[0], "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    with open(image_files[1], "rb") as image_file:
        encoded_string1 = base64.b64encode(image_file.read())
    st.markdown(
    """
    <style>
      .stApp {
          background-image: url(data:image/png;base64,"""+encoded_string.decode()+""");
          background-size: cover;
      }
      .css-6qob1r.e1fqkh3o3 {
        background-image: url(data:image/png;base64,"""+encoded_string1.decode()+""");
        background-size: cover;
        background-repeat: no-repeat;
      }
    </style>"""
    ,
    unsafe_allow_html=True
    )
add_bg_from_local([r'C:\Users\smartuse\OneDrive\Desktop\new\medical.jpg', r'C:\Users\smartuse\OneDrive\Desktop\new\waves.jpg'])


class SVM_classifier():

    # initiating the hyperparameters
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):

        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    # fitting the dataset to SVM Classifier
    def fit(self, X, Y):

        # m  --> number of Data points --> number of rows
        # n  --> number of input features --> number of columns
        self.m, self.n = X.shape

        # initiating the weight value and bias value

        self.w = np.zeros(self.n)

        self.b = 0

        self.X = X

        self.Y = Y

        # implementing Gradient Descent algorithm for Optimization

        for i in range(self.no_of_iterations):
            self.update_weights()

    # function for updating the weight and bias value
    def update_weights(self):

        # label encoding
        y_label = np.where(self.Y <= 0, -1, 1)

        # gradients ( dw, db)
        for index, x_i in enumerate(self.X):

            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

            if (condition == True):

                dw = 2 * self.lambda_parameter * self.w
                db = 0

            else:

                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
                db = y_label[index]

            self.w = self.w - self.learning_rate * dw

            self.b = self.b - self.learning_rate * db

    # predict the label for a given input value
    def predict(self, X):

        output = np.dot(X, self.w) - self.b

        predicted_labels = np.sign(output)

        y_hat = np.where(predicted_labels <= -1, 0, 1)

        return y_hat


class Logistic_Regression():

    # declaring learning rate & number of iterations (Hyperparametes)
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    # fit function to train the model with dataset
    def fit(self, X, Y):
        # number of data points in the dataset (number of rows)  -->  m
        # number of input features in the dataset (number of columns)  --> n
        self.m, self.n = X.shape

        # initiating weight & bias value

        self.w = np.zeros(self.n)

        self.b = 0

        self.X = X

        self.Y = Y

        # implementing Gradient Descent for Optimization

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        # Y_hat formula (sigmoid function)

        Y_hat = 1 / (1 + np.exp(- (self.X.dot(self.w) + self.b)))

        # derivaties

        dw = (1 / self.m) * np.dot(self.X.T, (Y_hat - self.Y))

        db = (1 / self.m) * np.sum(Y_hat - self.Y)

        # updating the weights & bias using gradient descent

        self.w = self.w - self.learning_rate * dw

        self.b = self.b - self.learning_rate * db

    # Sigmoid Equation & Decision Boundary

    def predict(self, X):
        Y_pred = 1 / (1 + np.exp(- (X.dot(self.w) + self.b)))
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred


# loading the saved models

diabetesmanual_model = pickle.load(open(r'C:\Users\smartuse\OneDrive\Desktop\new\saved models\diabetesmanual_model.sav', 'rb'))
diabetesmanual_model_scaler = pickle.load(open(r'C:\Users\smartuse\OneDrive\Desktop\new\saved models\diabetesmanual_model_scaler.sav', 'rb'))
heartmanual_model = pickle.load(open(r'C:\Users\smartuse\OneDrive\Desktop\new\saved models\heartmanual_model.sav', 'rb'))
heartmanual_model_scaler = pickle.load(open(r'C:\Users\smartuse\OneDrive\Desktop\new\saved models\heartmanual_model_scaler.sav', 'rb'))
parkinsonmanual_model = pickle.load(open(r'C:\Users\smartuse\OneDrive\Desktop\new\saved models\parkinsonmanual_model.sav', 'rb'))
parkinsonmanual_model_scaler = pickle.load(open(r'C:\Users\smartuse\OneDrive\Desktop\new\saved models\parkinsonmanual_model_scaler.sav', 'rb'))

diabetes_cols = ['Cholesterol', 'Glucose', 'HDL Chol', 'Chol/HDL ratio', 'Age', 'Gender',
                 'Height', 'Weight', 'BMI', 'Systolic BP', 'Diastolic BP', 'waist',
                 'hip', 'Waist/hip ratio']
heart_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
              'thal']

parkinson_cols = ['fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs', 'RAP', 'PPQ', 'DDP',
                  'Shimmer', 'Shimmer_dB', 'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE',
                  'DFA', 'spread1', 'spread2', 'D2', 'PPE']
# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    # col1, col2, col3 = st.columns(3)


    Age =  st.number_input('Age', min_value=0, max_value=100, step=1)

    Gender = st.radio("Select your gender", ["Male", "Female"])

    if Gender:
        if Gender == "Male":
            st.write("You selected Male")
        elif Gender == "Female":
            st.write("You selected Female")



    Cholesterol = st.number_input('Cholesterol',min_value=50, max_value=999, step=1)
    HDL_Chol = st.number_input('HDL Chol',min_value=1, max_value=500, step=1)
    Chol_HDL_ratio = st.number_input('Chol/HDL ratio',min_value=1, max_value=50, step=1)

    Glucose = st.number_input('Glucose Level',min_value=1, max_value=500, step=1)

    Systolic_BP = st.number_input('Systolic BP',min_value=50, max_value=500, step=1)
    Diastolic_BP = st.number_input('Diastolic BP',min_value=20, max_value=500, step=1)

    Height = st.number_input('Height',min_value=20, max_value=250, step=1)
    Weight = st.number_input('Weight',min_value=10, max_value=500, step=1)
    BMI = st.number_input('BMI value',min_value=10, max_value=80, step=1)

    waist = st.number_input('Waist',min_value=10, max_value=100, step=1)
    hip = st.number_input('Hip',min_value=10, max_value=100, step=1)
    Waist_hip_ratio = st.number_input('Waist/hip ratio',min_value=0, max_value=5, step=1)

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        data = pd.DataFrame([[Cholesterol, Glucose, HDL_Chol, Chol_HDL_ratio, Age, 1 if Gender == 'Male' else 0,
                              Height, Weight, BMI, Systolic_BP, Diastolic_BP, waist,
                              hip, Waist_hip_ratio]], columns=diabetes_cols)

        # standardize the input data
        std_data = diabetesmanual_model_scaler.transform(data)

        diab_prediction = diabetesmanual_model.predict(std_data)

        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)
    




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):

    # page title
    st.title('Heart Disease Prediction using ML')

    # col1, col2, col3 = st.columns(3)

    age = st.number_input('Age',min_value=1, max_value=110, step=1)

    sex = st.radio('Sex', ('Male', 'Female'))

    cp = st.selectbox('Chest Pain types',('0','1','2','3'))

    trestbps = st.number_input('Resting Blood Pressure',min_value=80, max_value=200, step=1)

    chol = st.number_input('Serum Cholestoral in mg/dl',min_value=100,max_value=500,step=1)

    fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl',min_value=0, max_value=5, step=1)

    restecg = st.number_input('Resting Electrocardiographic results',min_value=0, max_value=5, step=1)

    thalach = st.number_input('Maximum Heart Rate achieved',min_value=50, max_value=300, step=1)

    exang = st.number_input('Exercise Induced Angina',min_value=0, max_value=5, step=1)

    oldpeak = st.number_input('ST depression induced by exercise',min_value=0, max_value=10, step=1)

    slope = st.number_input('Slope of the peak exercise ST segment',min_value=0, max_value=5, step=1)

    ca = st.number_input('Major vessels colored by flourosopy',min_value=0, max_value=5, step=1)

    thal = st.selectbox('Thal', ('Normal', 'Fixed Defect', 'Reversable Defect'))

    thal = 0 if thal == 'Normal' else 1 if thal == 'Fixed Defect' else 2

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        data = pd.DataFrame(
            [[age, 1 if sex == 'Male' else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca,
              thal]], columns=heart_cols)

        # standardize the input data
        std_data = heartmanual_model_scaler.transform(data)

        heart_prediction = heartmanual_model.predict(std_data)

        if (heart_prediction[0] == 1):
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)


# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    # col1, col2, col3, col4, col5 = st.columns(5)

    fo = st.number_input('MDVP:Fo(Hz)',min_value=50, max_value=300, step=1)

    fhi = st.number_input('MDVP:Fhi(Hz)',min_value=50, max_value=300, step=1)

    flo = st.number_input('MDVP:Flo(Hz)',min_value=150, max_value=300, step=1)

    Jitter_percent = st.number_input('MDVP:Jitter(%)',min_value=0, max_value=5, step=1)

    Jitter_Abs = st.number_input('MDVP:Jitter(Abs)',min_value=0, max_value=5, step=1)

    RAP = st.number_input('MDVP:RAP',min_value=0, max_value=5, step=1)

    PPQ = st.number_input('MDVP:PPQ',min_value=0, max_value=5, step=1)

    DDP = st.number_input('Jitter:DDP',min_value=0, max_value=5, step=1)

    Shimmer = st.number_input('MDVP:Shimmer',min_value=0, max_value=5, step=1)

    Shimmer_dB = st.number_input('MDVP:Shimmer(dB)',min_value=0, max_value=5, step=1)

    APQ3 = st.number_input('Shimmer:APQ3',min_value=0, max_value=5, step=1)

    APQ5 = st.number_input('Shimmer:APQ5',min_value=0, max_value=5, step=1)

    APQ = st.number_input('MDVP:APQ',min_value=0, max_value=5, step=1)

    DDA = st.number_input('Shimmer:DDA',min_value=0, max_value=5, step=1)

    NHR = st.number_input('NHR',min_value=0, max_value=5, step=1)

    HNR = st.number_input('HNR',min_value=10, max_value=50, step=1)

    RPDE = st.number_input('RPDE',min_value=0, max_value=5, step=1)

    DFA = st.number_input('DFA',min_value=0, max_value=5, step=1)

    spread1 = st.number_input('spread1',min_value=-10, max_value=5, step=1)

    spread2 = st.number_input('spread2',min_value=0, max_value=5, step=1)

    D2 = st.number_input('D2',min_value=0, max_value=5, step=1)

    PPE = st.number_input('PPE',min_value=0, max_value=5, step=1)

    # code for Prediction
    parkinsons_diagnosis = ''
    data = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
            Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE,
            DFA, spread1, spread2, D2, PPE]

    data = np.asarray(data)
    data = data.reshape(1, -1)

    # creating a button for Prediction
    if st.button("Parkinson's Test Result"):

        # standardize the input data
        std_data = parkinsonmanual_model_scaler.transform(data)

        parkinsons_prediction = parkinsonmanual_model.predict(std_data)

        if (parkinsons_prediction[0] == 1):
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

#st.markdown("Click [here](https://discuss.streamlit.io/t/hyperlink-in-streamlit-without-markdown/7046/10) for more .")

if st.button("Reset"):
    input_sms=""
