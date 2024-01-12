
import cv2
import cvzone
from fastapi import File, FastAPI, UploadFile, WebSocket,WebSocketDisconnect,Form,Query, APIRouter,Depends,status,BackgroundTasks
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
from ultralytics import YOLO
import math
import os
from backend.repository import data
import uuid
from typing import List
from sqlalchemy.orm import Session
from db.datebase import get_db

router = APIRouter(
    prefix="/model",
    tags=["Models"]
)
 
#model = YOLO('yolov8n.pt')

@router.get("/")
def read_root():
    return {"message":"Welcome from the API"}

@router.post("/detect_logo")
async def get_predict(detect_name: str = Form(...), model_id: str = Form(...), user_id: str = Form(...), files: list[UploadFile] = None, db:Session = Depends(get_db)):
    status = await data.get_predict(detect_name, model_id, user_id, files, db)

    return status

def redimension(image):

    height, width = int(image.shape[0]), int(image.shape[1])
    new_width = int((640 / height) * width)
    resized_image = cv2.resize(image, (new_width, 640), interpolation=cv2.INTER_AREA)

    return  resized_image

@router.post("/training/{id_annotation}/{id_usuario}/{model_name}")
#async def send_notification(background_tasks: BackgroundTasks,id_annotation: str= Form(...),id_usuario: str= Form(...),model_name: str= Form(...), db:Session = Depends(get_db)):
async def send_notification(id_annotation: str,id_usuario: str,model_name: str, background_tasks: BackgroundTasks,db:Session = Depends(get_db)):
    background_tasks.add_task(data.run_train,model_name,id_annotation,id_usuario,db)
    return {"message": "Notification sent in the background"}


@router.post("/load_image")
def loader_dataset(dataset_name: str = Form(...), labels: str = Form(...), user_id: str = Form(...),  files: list[UploadFile] = None,db:Session = Depends(get_db)):

    status = data.cargar_dataset(dataset_name, labels, files,user_id,db)

    return status

@router.post("/get_dataset",status_code=status.HTTP_200_OK)
def get_dataset(db:Session = Depends(get_db)):

    status = data.get_all_dataset(db)

    return status

@router.post("/save_annotation")
def loader_dataset(annotation_name: str = Form(...), labels: str = Form(...), user_id: str = Form(...),  files: list[UploadFile] = None,db:Session = Depends(get_db)):

    status = data.cargar_annotation(annotation_name, labels, files,user_id,db)

    return status

@router.post("/get_annotation",status_code=status.HTTP_200_OK)
def get_annotation(db:Session = Depends(get_db)):

    status = data.get_all_annotation(db)

    return status

@router.post("/get_models",status_code=status.HTTP_200_OK)
def get_models(db:Session = Depends(get_db)):

    status = data.get_all_models(db)

    return status


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
@router.websocket("/airquality")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            async for resp in data.logGenerator():
                await websocket.send_json(resp)
            break         
    except WebSocketDisconnect:
        manager.disconnect(websocket)