import streamlit as st
import io
import sys

# Function that will generate the output to be shown
def generate_output():
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    some_python_code()  # Your python code goes here
    output = new_stdout.getvalue()
    sys.stdout = old_stdout
    return output

# Python code to be executed on button press
def some_python_code():
    print("This is a sample print statement.")

# Streamlit app interface
st.title('Streamlit Predict Button')

if st.button('Predict'):
    output = generate_output()
    st.text(output)
