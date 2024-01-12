import requests
import aiohttp
import json
import asyncio

class API:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.base_headers = {
                    "Authorization":f"Bearer {token}"
                }
            
    def is_logged_in(self):

        try:
            status = False
            status = requests.get(self.base_url + "/user/is_logged_in",headers=self.base_headers).status_code == 200
        except:
            return status
        return status

    # def add_employee(self, data):
        
    #     try:
    #         # data = {
    #         #     "name": name,
    #         #     "date_of_birth": dob,
    #         #     "paygrade_id": paygrade
    #         # }

    #         #response = requests.post(self.base_url + "/user", json=data, headers=self.base_headers)  
    #         url_post = f{self.base_url}  + "/user"        
    #         async with aiohttp.ClientSession as session:
    #             response = await session.request(method="POST", url=url_post, json=data)
    #             response_json = await response.json()
    #             if "respuesta" un response_json:
    #                 msj = True
    #             else
    #                 msj = False
    #         if response.status_code == 200:
    #             return True
    #     except:
    #         return False
            
    # def get_employees(self):
    #     try:
    #         response = requests.get(self.base_url + "/employees", headers=self.base_headers)
    #         return response.json()['data']
    #     except:
    #         return None

    def login( self,username, password):
        try:
           
            token = None
            url_post = self.base_url        
            # async with aiohttp.ClientSession() as session:
            #     response = await session.request(method="POST", url=url_post, data={
            #     "username": username,
            #     "password": password}
            #     )
            response = requests.post(url_post+"/login/",data={
                "username": username,
                "password": password}
                )
            assert response.status_code == 200
            assert response.json()["token_type"] == "bearer"


            if 'access_token' not in response.json():
                msj = "Usuario o contraseña incorrecto"
            else:
                body = response.json()
                
                token = body.get("access_token")
            return token
        except:
            return token


    def get_user_from_username( self,username):
        try:
           
            token = None
            url_post = self.base_url        
            response = requests.get(url_post+"/user/get_user_from_username/"+username)
            assert response.status_code == 200

            return response.json()#["role_id"] 
        except:
            return token

    def save_dataset(self,data,files):

        try:
            status = False

            status = requests.post(self.base_url + "/model/load_image",headers=self.base_headers,data=data ,files=files).status_code == 200
        except:
            return status
        return status

    async def detect_logo(self,data,files):

        try:

            response = requests.post(self.base_url + "/model/detect_logo",headers=self.base_headers,data=data ,files=files)
            if response.status_code == 200:
                return response.json()
            else:
                data = False
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            return None
        return None



    def get_dataset(self):

        try:
            response = False

            #response = requests.post(self.base_url + "/model/get_dataset",headers=self.base_headers)
            response = requests.post(self.base_url + "/model/get_dataset")
            
            assert response.status_code == 200

        except:
            return response
        return response.json()
    
    def save_annotation(self,data,files):
            try:
                status = False
                status = requests.post(self.base_url + "/model/save_annotation",headers=self.base_headers,data=data ,files=files).status_code == 200
            except:
                return status
            return status
    
    def get_annotation(self):
        try:
            response = False
            #response = requests.post(self.base_url + "/model/get_dataset",headers=self.base_headers)
            response = requests.post(self.base_url + "/model/get_annotation")
            
            assert response.status_code == 200
        except:
            return response
        return response.json()

    def get_models(self):
        try:
            response = False
            #response = requests.post(self.base_url + "/model/get_dataset",headers=self.base_headers)
            response = requests.post(self.base_url + "/model/get_models")
            
            assert response.status_code == 200
        except:
            return response
        return response.json()

    def run_training(self, model_name, id_annotation, id_usuario):

        try:
            status = False
            #data = '2'
            string_prueba = f"/model/training/{id_annotation}/{id_usuario}/{model_name}"
            status = requests.post(self.base_url + f"/model/training/{id_annotation}/{id_usuario}/{model_name}")
        except:
            return status
        return status

    async def consumer_airquality(self,my_bar):
        WS_CONN = "ws://localhost:8000/model/airquality"
        async with aiohttp.ClientSession(trust_env=True) as session:
            #status.subheader(f"Connecting to {WS_CONN}")
            async with session.ws_connect(WS_CONN) as websocket:
                #status.subheader(f"Connected to: {WS_CONN}")
                async for message in websocket:
                    json_data = json.loads(message.data)
                    print( type(message.type))
                    if isinstance(json_data, list) and len(json_data) > 0:
                            # Access the first element of the JSON array
                            first_element = json_data[0]
                            my_bar.progress((int(first_element) + 1 )* 2, text="Operacion en proceso. Por favor espere.")
                            print(first_element)
                            #st.write(first_element)
        my_bar.progress(100,text="Operacion en proceso. Por favor espere.")