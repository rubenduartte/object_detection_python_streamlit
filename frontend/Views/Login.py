import streamlit as st
from typing import Callable
class Login:
    def __init__(self, on_login: Callable[[str, str], bool]):
        
        with st.form("my_form"):  
                st.header("Ingreso")        
                username = st.text_input("Username")
                password = st.text_input("Password",type="password")
                submit = st.form_submit_button("Login")
                if submit:
                    success = on_login(username, password)
                    if success:
                        st.success("Login successful")
                        st.experimental_rerun()
                    else:
                        st.error("Incorrect username and password combination")