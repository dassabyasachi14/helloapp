import streamlit as st

st.write("# Streamlit Calculator")
number1 = st.number_input("First Number")
number2 = st.number_input("Second Number")
num3=number1+number2
st.write("Output is",num3)
