#import cv2
from fastapi import File, FastAPI, WebSocket,WebSocketDisconnect,BackgroundTasks
#from sh import tail
from typing import List
from ultralytics import YOLO

import sys
from io import StringIO
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor
import csv
import time
import os
import json

#logging.basicConfig(level=logging.DEBUG,format='%(threadName)s: %(message)s')

app = FastAPI()

def run_train():
    model = YOLO('yolov8n.pt')
    model.train(data='C:\\Users\\Ruben\\Desktop\\sistema-logo\\model\\frames\\data.yaml', epochs=50, imgsz=255, device='cpu')


@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_train)
    return {"message": "Notification sent in the background"}





# # import numpy as np
# # from PIL import Image
# import uvicorn
# # import asyncio
# import os

# app=FastAPI()

# #real_path = os.path.realpath(__file__)
# #dir_path = os.path.dirname(real_path)
# #LOGFILE = f"{dir_path}/test.log"
# import time 

# is_training_running = False
# executor = ThreadPoolExecutor(max_workers=1)
# def runTrain():
#     #stdout_buffer = StringIO()
#     #sys.stdout =  open('output.txt', 'w')
#     model = YOLO('yolov8n.pt')
#     model.train(data='C:\\Users\\Ruben\\Desktop\\sistema-logo\\model\\frames\\data.yaml', epochs=1, imgsz=255, device='cpu')
#     #sys.stdout.close()
#     logging.info('Terminamos la tarea compleja!!\n')

#Espera que el archivo se cree primero para poder continuar
def wait_for_file(file_path, polling_interval=1):
    while not os.path.exists(file_path):
        time.sleep(polling_interval)

#Verifica la finalizacion del entrenamiento
def check_finished_train(): 
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    model_name = os.path.join(file_dir,  r'runs\detect\train\results.png')
    if os.path.exists(model_name):
        return True
    else:
        return False

#Lee el result del entrenamiento y devuelve los valores que se usa en el front para saber el proceso
async def logGenerator():

    file_dir = os.path.dirname(os.path.realpath('__file__'))
    file_name = os.path.join(file_dir,  r'runs\detect\train\results.csv')
    wait_for_file(file_name) #Espera hasta que se cree el registro para poder continuar
    
    csv_reader = csv.reader(FileTailer(open(file_name)))
    is_first_row = True
    for row in csv_reader:
        if is_first_row:
                is_first_row = False
                continue
        cleaned_row = [item.strip() for item in row]
        yield cleaned_row
        
        

#clase que actua como tail lee y retorna el ultimo registro agregado al excel
class FileTailer(object):
    def __init__(self, file, delay=0.1):
        self.file = file
        self.delay = delay
    def __iter__(self):
        while True:
            if check_finished_train(): #constantemente esta verificando si hay nuevas lineas y tambien si ya se creo el archivo
                break
            where = self.file.tell()
            line = self.file.readline()
            if line and line.endswith('\n'): # only emit full lines
                yield line
            else:                            # for a partial line, pause and back up
                time.sleep(self.delay)       # ...not actually a recommended approach.
                self.file.seek(where)



class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()



#socket de lectura del archivo csv
@app.websocket("/airquality")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            async for data in logGenerator():
                await websocket.send_json(data)
                # if await check_for_best_model():
                #     await websocket.close()
                    
                #     manager.disconnect(websocket)
                #     break
            break         
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# # @app.post("/predict")
# # def get_predict(file: UploadFile = File(...)):
# #     image = np.array(Image.open(file.file))
# #     # model = config.STYLES[style]
# #     resized = redimension( image)
# #     # name = f"/storage/{str(uuid.uuid4())}.jpg"
# #     # cv2.imwrite(name, output)
# #     return {"filename": "Hola"}


# # def redimension( image):

# #     height, width = int(image.shape[0]), int(image.shape[1])
# #     new_width = int((640 / height) * width)
# #     resized_image = cv2.resize(image, (new_width, 640), interpolation=cv2.INTER_AREA)

# #     return  resized_image

