import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


st.title('shubham ahire')
st.title('Loan Prediction')

st.sidebar.header('User Input Parameter')

def user_input_features():
    Married = st.sidebar.selectbox("Enter Marriage Status",('Yes','No'))
    Education = st.sidebar.selectbox("Education Qualification",('Graduate','Not Graduate'))
    ApplicantIncome = st.sidebar.number_input("Input Salary")
    CoapplicantIncome = st.sidebar.number_input("Co-Applicant Income")
    LoanAmount = st.sidebar.number_input("Input Loan Amount")
    Credit_History = st.sidebar.selectbox("Credit History",('Yes','No'))
    Property_Area = st.sidebar.selectbox("Property Area",('Rural','Semiurban','Urban'))
    data = {
        'Married':Married,
        'Education':Education,
        'ApplicantIncome':ApplicantIncome,
        'CoapplicantIncome':CoapplicantIncome,
        'LoanAmount':LoanAmount,
        'Credit_History':Credit_History,
        'Property_Area':Property_Area,
    }
    features = pd.DataFrame(data,index = [0])
    return features

test = user_input_features()
st.subheader('User Input Parameter')
st.write(test)

#Encoding the test values in dataset
test['Married'].replace({'Yes':1, 'No':0}, inplace=True)
test['Credit_History'].replace({'Yes':1, 'No':0}, inplace=True)
test['Education'].replace({'Graduate':1, 'Not Graduate':0}, inplace=True)
test['Property_Area'].replace({'Rural':0, 'Semiurban':1,'Urban':2}, inplace=True)
st.write(test)

#Handeling Missing values
train_imp = pd.read_csv(r'C:\Users\user\Documents\Loan-Prediction-Using-Streamlit-master\train.csv')
train_imp.drop(['Loan_ID','Gender','Dependents','Self_Employed','Loan_Amount_Term'],axis=1, inplace=True)

cat_null = ['Married','Credit_History','Education','Property_Area']
con_null = ['ApplicantIncome','CoapplicantIncome','LoanAmount']

# Run the imputer with a simple Random Forest estimator
imp = IterativeImputer(RandomForestRegressor(n_estimators=5), max_iter=5, random_state=1)
to_train = con_null
#perform filling
train_imp[to_train] = pd.DataFrame(imp.fit_transform(train_imp[to_train]), columns=to_train)

# Imputer object using the mean strategy and
# missing_values type for imputation
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import  RandomForestClassifier
train_imp[cat_null] = train_imp[cat_null].apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]),index=series[series.notnull()].index))
imp_cat = IterativeImputer(estimator=RandomForestClassifier(),initial_strategy='most_frequent',max_iter=10, random_state=0)
train_imp[cat_null] = imp_cat.fit_transform(train_imp[cat_null])

#Handeling Outliers
for x in ['ApplicantIncome','CoapplicantIncome','LoanAmount']:
    q75,q25 = np.percentile(train_imp.loc[:,x],[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    train_imp.loc[train_imp[x] < min,x] = min
    train_imp.loc[train_imp[x] > max,x] = max


#model building
x = train_imp.drop('Loan_Status',1)
y = train_imp.Loan_Status
model = BaggingClassifier()
model.fit(x,y)

st.write(x.head())

#prediction of model
prediction = model.predict(test)
prediction_proba = model.predict_proba(test)

st.subheader('Prediction Results')
st.write('Yes' if prediction_proba[0][1] > 0.3 else 'No')

st.subheader('Prediction Probablity')
st.write(prediction_proba)
print(train_imp)
