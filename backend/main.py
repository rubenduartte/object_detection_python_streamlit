import cv2
import cvzone
from fastapi import File, FastAPI, UploadFile,HTTPException,Form,Query
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import uvicorn
from ultralytics import YOLO
import math
import os

import uuid
from datetime import datetime
from db.datebase import Base, engine
from backend.routers import user, auth,dnn

# def create_table():
#     Base.metadata.create_all(bind=engine)

# create_table()

app=FastAPI()

app.include_router(dnn.router)
app.include_router(user.router)
app.include_router(auth.router)


if __name__=="__main__":
    uvicorn.run("main:app",port=8000, reload=True)
# model = YOLO('yolov8n.pt')

    

# @app.get("/")
# def read_root():
#     return {"message":"Welcome from the API"}


# @app.post("/predict")
# async def get_predict(files: list[UploadFile]):

#     model_name= 'predict'
#     path = os.getcwd()
#     output_images_path = path + "/" + model_name

#     if not os.path.exists(output_images_path):
#         os.makedirs(output_images_path)
    
#     count = 0

#     for file in files:
#         model = YOLO('yolov8n.pt')
#         #image = np.array(Image.open(file.file))

#         image =await cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

#         if image is None:
#             continue
#         if file.filename.split(".")[-1] not in ["jpg","jpeg", "png"]:
#             continue
        

#         resized = redimension( image)
#         # name = f"/storage/{str(uuid.uuid4())}.jpg"
#         # cv2.imwrite(name, output)
#         results =  model(resized, stream=True,save=True)
#         #cv2.imshow("result", res_plotted)
#         for r in results:
#             boxes = r.boxes
#             prueba = r.probs
#             res_plotted = r.plot(probs=True)
#             #cv2.imshow("result", res_plotted)
#             #cv2.waitKey(1)
#             for box in boxes:
                
#                 # Bounding Box
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 #cv2.rectangle(img,(x1, y1),(x2,y2),(255,0,255),3)
                
#                 w, h = x2 -x1, y2 -y1
                
#                 #Confianza
#                 conf = math.ceil((box.conf[0] * 100 )) / 100
#                 #Class Name
#                 cls = int(box.cls[0])
#                 myColor= (0, 255, 0)
#                 currentClass = r.names[cls]
#                 if conf > 0.3:
#                     cvzone.cornerRect(resized, (x1, y1, w, h),l=8,rt=5)
#                     #currenArray = np.array([x1, y1, x2, y2, conf])
#                     cvzone.putTextRect(resized, f'{r.names[cls]} {conf}', 
#                                     (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
#                                     colorT=(255,255,255),colorR=myColor, offset=5)
                    
#                     #cv2.rectangle(resized,(x1,y1), (x2, y2), myColor, 3)
#         cv2.imwrite(output_images_path + "/image" + str(count) + ".jpg", resized)
#         count += 1
#     return {"filename": "Hola"}

# def redimension( image):

#     height, width = int(image.shape[0]), int(image.shape[1])
#     new_width = int((640 / height) * width)
#     resized_image = cv2.resize(image, (new_width, 640), interpolation=cv2.INTER_AREA)

#     return  resized_image

# @app.post("/training")
# async def training():
#     #model = YOLO('yolov8n.pt')
#     await model.train(data='C:\\Users\\Ruben\\Desktop\\sistema-logo\\model\\frames\\data.yaml', epochs=2, imgsz=255, device='cpu')
#     return {"message":"Welcome from the API"}


# @app.post("/load_image")
# def loader_dataset(dataset_name: str = Form(...), labels: list[str] = Form(...), files: list[UploadFile] = None):
      
#     dataset_path = create_dataset_folder(dataset_name)


#     for count, file in enumerate(files, start=1):
#         try:
#             nparr = np.asarray(bytearray(file.file.read()), dtype="uint8")
#             #nparr = np.fromstring(bytearray(file.read()), np.uint8)
#             image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
#             # Resto del código...
#         except Exception as e:
#             print(f"Error decoding image: {e}")

#         resized = redimension( image)

#          # Obtén la extensión del archivo
#         file_extension = os.path.splitext(file.filename)[1]

#         # Genera el nuevo nombre del archivo
#         new_file_name = f"IMAGE-{count}{file_extension}"
#         # Genera la ruta completa del archivo
#         image_path = os.path.join(dataset_path, new_file_name)

#         # Guarda la imagen en la ruta especificada
#         cv2.imwrite(image_path, resized)

#     # Guardar classes.txt
#     save_classes_txt(dataset_path, labels)


#     return {"message":"Welcome from the API"}



# def create_dataset_folder(dataset_name):
#     # Obtener el año y mes actual
#     current_year = datetime.now().strftime("%Y")
#     current_month = datetime.now().strftime("%Y%m")
    
#     path= os.path.join(os.getcwd(),'dataset')
    
#     if not os.path.exists(path):
#         os.makedirs(path)
#     # Crear un UUID único
#     unique_id = str(uuid.uuid4())
    
#     # Crear el nombre de la carpeta final
#     folder_name = f"{current_year}/{current_month}/{unique_id}_{dataset_name}"
    
#     # Construir la ruta completa
#     full_path = os.path.join(path, folder_name)
    
#     # Verificar si la carpeta ya existe
#     if not os.path.exists(full_path):
#         os.makedirs(full_path)
    
#     return full_path

# def save_classes_txt(dataset_path: str, labels: list[str]):
#     l = labels[0].split(",")
#     classes_txt_path = os.path.join(dataset_path, "classes.txt")

#     with open(classes_txt_path, "w") as f:
#         for label in l:
#             f.write(f"{label}\n")