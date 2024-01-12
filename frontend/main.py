import streamlit as st
from Views import Login, Board
from API import API
import extra_streamlit_components as stx
import base64, json
import time

def manage_login(username, password):
    token = api.login(username, password)

    cookie_manager.set("access_token", token)
    time.sleep(3)
    return token is not None


def get_username_from_token(auth_token):
    b64 = str(auth_token).split(".")[1]
    b64 = b64 + "=" * (4 - (len(b64) % 4))
    data = base64.b64decode(b64).decode("utf8")
    username = json.loads(data)['sub']
    return username
st.set_page_config(page_title="Sistema Reconocimiento Logo")

# st.markdown("""
# <style>
# .css-czk5ss.e16jpq800
# {
#     visibility: hidden;
# }                      
# </style>
            
#             """, unsafe_allow_html=True)

cookie_manager = stx.CookieManager()
authentication_token = cookie_manager.get("access_token")
api = API("http://127.0.0.1:8000", authentication_token)
if api.is_logged_in():
     Board.Board(get_username_from_token(authentication_token),api)
else:
    Login.Login(manage_login)

