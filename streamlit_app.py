import streamlit as st
import io
import sys

# Function that will generate the output to be shown
def generate_output(user_input):
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    some_python_code(user_input)  # Your python code goes here
    output = new_stdout.getvalue()
    sys.stdout = old_stdout
    return output

# Python code to be executed on button press
def some_python_code(user_input):
    print(f"This is the user input: {user_input}")

# Streamlit app interface
st.title('Streamlit Predict Button')

# Text box for user input
user_input = st.text_input('Enter a string:')

if st.button('Predict'):
    if user_input:  # Execute only if user has provided some input
        output = generate_output(user_input)
        st.text(output)
    else:
        st.warning('Please input a string.')
