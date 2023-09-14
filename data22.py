import streamlit as st
import pandas as pd
import numpy as np
import time
import PIL
import sklearn
from PIL import Image
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


img = Image.open("streamlit.jpg")
st.image(img, width=300)




st.success("Success")
st.info("Information")
st.warning("Warning")
st.error("Error")
exp = ZeroDivisionError("Trying to divide by Zero")
st.exception(exp)


if st.checkbox("Show/Hide"):
    st.text("Showing the widget")

status = st.radio("Gender: ",('Male','Female'))

if (status == 'Male'):
    st.success("Male")
else:
    st.info("Female")

hobby = st.selectbox("Hobbies: ",['Dancing', 'Reading', 'Sports'])

st.write("Your hobby is : ", hobby)

hobbies = st.multiselect("Hobbies: ",['Dancing', 'Reading', 'Sports'])

st.write("You selected", len(hobbies), "hobbies")

st.button("Click me for no reason")


if(st.button("About")):
    st.success("Welcome to this page")



name = st.text_input("Enter your name", "")

if(st.button("Submit")):
    result = name.title()
    st.success(result)




level = st.slider("Select the level",1,100)

st.text('Selected: {}'.format(level))


st.sidebar.header('Sidebar heading')

def user_selection_funtion():
    Selection1 = st.sidebar.slider('**_Selection1_**',0,100,10,10)
    Selection2 = st.sidebar.slider('_Selection2_',0,10,1,1)
    Selection3 = st.sidebar.slider('**Selection3**',0,15,None,1)
    Selection4 = st.sidebar.slider('**Selection4**',0,100,10,1)

    data = {'Selection1':Selection1,
            'Selection2':Selection2,
            'Selection3':Selection3,
            'Selection4':Selection4}
    
    selection = pd.DataFrame(data,index=[0])
    return selection

df = user_selection_funtion()

st.subheader('**User Slection**')
st.write(df)


iris = datasets.load_iris()
digits = datasets.load_digits()

X = iris.data
Y = iris.target
A = digits.data
B = digits.target

print(X)
print(Y)
print(A)
print(B)

clf = RandomForestClassifier()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Corresonding index number')
st.write(iris.target_names)

