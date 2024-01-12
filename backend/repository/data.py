from sqlalchemy.orm import Session
from db import models
from fastapi import HTTPException,status,Response
from fastapi.responses import FileResponse,StreamingResponse,JSONResponse
from backend.hashing import Hash
import os
import base64
from datetime import datetime
import numpy as np
import uuid
from ultralytics import YOLO
import time
import csv
import shutil
import yaml
import cv2
import cvzone
from ultralytics import YOLO
import math
from backend import modulo_aument
import io

TAMANO = int(640)
conf_train = float(0.1)
conf_detect = float(0.38)

def cargar_dataset(dataset_name, labels, files,user_id, db:Session):

    dataset_path = create_folder(dataset_name, 1)

    for count, file in enumerate(files, start=1):
        try:
            nparr = np.asarray(bytearray(file.file.read()), dtype="uint8")
            image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            # Resto del código...
        except Exception as e:
            print(f"Error decoding image: {e}")
        resized = redimension(image)

         # Obtén la extensión del archivo
        file_extension = os.path.splitext(file.filename)[1]

        # Genera el nuevo nombre del archivo
        new_file_name = f"IMAGE-{count}{file_extension}"
        # Genera la ruta completa del archivo
        image_path = os.path.join(dataset_path, new_file_name)

        # Guarda la imagen en la ruta especificada
        cv2.imwrite(image_path, resized)
        # Guardar classes.txt
        save_classes_txt(dataset_path, labels)

    try:
        new_dataset = models.Dataset(
            usuario_id= user_id,
            dataset_name=dataset_name,
            path=dataset_path,
            labels=labels
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Error creando Dataset {e}"
    )
    

def create_folder(name, type):
    # Obtener el año y mes actual
    current_year = datetime.now().strftime("%Y")
    current_month = datetime.now().strftime("%Y%m")
    
    if type == 1:
        path= os.path.join(os.getcwd(),'dataset')
    elif type == 2:
        path= os.path.join(os.getcwd(),'annotation')
    elif type == 3:
        path= os.path.join(os.getcwd(),'models')        
    else:
        path = os.path.join(os.getcwd(),'detection')


    if not os.path.exists(path):
        os.makedirs(path)
    # Crear un UUID único
    unique_id = str(uuid.uuid4())
    
    # Crear el nombre de la carpeta final
    folder_name = f"{current_year}/{current_month}/{unique_id}_{name}"
    
    # Construir la ruta completa
    full_path = os.path.join(path, folder_name)
    
    # Verificar si la carpeta ya existe
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    # Si el tipo 3, crea la estructura de carpetas dentro de "full_path"
    if type == 2:
        os.makedirs(os.path.join(full_path, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(full_path, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(full_path, 'val', 'images'), exist_ok=True)
        os.makedirs(os.path.join(full_path, 'val', 'labels'), exist_ok=True)
    
    return full_path

def save_classes_txt(dataset_path: str, labels: str):
    l = labels.split(",")
    classes_txt_path = os.path.join(dataset_path, "classes.txt")

    with open(classes_txt_path, "w") as f:
        for label in l:
            f.write(f"{label}\n")
        f.close()

# def save_annotation_txt(classes_txt_path: str, file_content):
#     #classes_txt_path = os.path.join(annotation_path, filename)

#     with open(classes_txt_path, "w") as f:
#         f.write(file_content.read())
#         f.close()

def save_annotation_txt(annotation_path: str, fil:str , file_content : bytes):
    classes_txt_path = os.path.join(annotation_path, 'train', 'labels', fil)

    with open(classes_txt_path, "wb") as f:
        f.write(file_content)
        f.close()

    classes_txt_path = os.path.join(annotation_path, 'val', 'labels', fil)

    with open(classes_txt_path, "wb") as f:
        f.write(file_content)
        f.close()
    
def create_data_yaml(annotation_path, train_path, val_path, class_names):
    # Dividir la cadena de clases separadas por comas en una lista
    class_names = [f"{cls.strip()}" for cls in class_names.split(',')]
    print(class_names)
    data = { 
        'train': os.path.join(os.getcwd(),"models","tmp","train","images"),  # Ruta completa de las imágenes de entrenamiento
        'val': os.path.join(os.getcwd(),"models","tmp","val","images"),        # Ruta completa de las imágenes de validación
        'names': {i: cls for i, cls in enumerate(class_names)}                                  # Nombres de las clases como una lista de cadenas
    }
    #data['names'] = {i: cls for i, cls in enumerate(class_names)}
    # Guardar el diccionario en un archivo YAML
    yaml_file_path = os.path.join(annotation_path, 'data.yaml')
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

def redimension( image,size = 255):

    height, width = int(image.shape[0]), int(image.shape[1])
    new_width = int((size / height) * width)
    resized_image = cv2.resize(image, (new_width, size), interpolation=cv2.INTER_AREA)

    return  resized_image

def get_all_dataset(db):
    dataset = db.query(models.Dataset).all()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe ningun Dataset"
        )
    
    return dataset

def cargar_annotation(annotation_name, labels, files,user_id, db:Session):

    annotation_path = create_folder(annotation_name, 2)
    train_images_path = os.path.join(annotation_path, 'train', 'images')
    val_images_path = os.path.join(annotation_path, 'val', 'images')
    # train_image = os.path.join(annotation_path, r'train\images')
    # val_image = os.path.join(annotation_path, r'val\images')

    #file_dir = os.path.dirname(os.path.realpath('__file__'))

    for count, file in enumerate(files, start=1):
        try:
            if file.content_type.startswith('image'):
                nparr = np.asarray(bytearray(file.file.read()), dtype="uint8")
                image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                #image_path = os.path.join(train_image, file.filename)
                cv2.imwrite(os.path.join(train_images_path, file.filename), image)
                cv2.imwrite(os.path.join(val_images_path, file.filename), image)
                #cv2.imwrite(image_path, image)
            else:
                #save_annotation_txt(annotation_path, file.filename,file.file.read())
                # print(type(file.file))
                # prueba = file.file
                # save_annotation_txt(os.path.join(annotation_path, 'train', 'labels', file.filename),prueba)
                # save_annotation_txt(os.path.join(annotation_path, 'val', 'labels', file.filename),prueba)

                save_annotation_txt(annotation_path, file.filename,file.file.read())

        except Exception as e:
            print(f"Error decoding file: {e}")

    #create_data_yaml(annotation_path, train_images_path, val_images_path, labels)
    create_data_yaml(annotation_path, train_images_path, val_images_path, labels)

    try:
        new_annotation = models.Annotation(
            usuario_id= user_id,
            annotation_name=annotation_name,
            path=annotation_path,
            labels=labels
        )
        db.add(new_annotation)
        db.commit()
        db.refresh(new_annotation)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Error creando Anotaciones {e}"
    )
    
def get_all_annotation(db):
    annotation = db.query(models.Annotation).all()

    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe ningun Anotacion"
        )
    
    return annotation

def get_all_models(db):
    models_gen = db.query(models.Model_Gen).all()

    if not models_gen:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe ningun Modelo"
        )
    
    return models_gen


def run_train(model_name,annotation_id, user_id, db):

    annotation = db.query(models.Annotation).filter(models.Annotation.id == annotation_id).first()

    move_annotation_tmp_for_train(annotation.path)

    data = os.path.join(os.getcwd(),"models","tmp",  "data.yaml") 
    labels = annotation.labels
    model_name = "prueba"
    create_train_tmp()

    imagespath = os.path.join(os.getcwd(),"models","tmp", "train", "images") 

    for filename in os.listdir(imagespath):

        image = readImage(filename,imagespath)
        resized = redimension(image)

        image_path = os.path.join(imagespath,filename)

        # Guarda la imagen en la ruta especificada
        cv2.imwrite(image_path, resized)


    modulo_aument.start(os.path.join(os.getcwd(),"models","tmp","train"))

    model = YOLO('yolov8n.pt')
    #data = 'C:\\Users\\Ruben\\Desktop\\sistema-logo\\annotation\\2023\\202309\\b5de83fa-97a8-4be7-b29e-6de1135f9e7b_prueba\\data.yaml'
    #model.train(data='C:\\Users\\Ruben\\Desktop\\sistema-logo\\model\\frames\\data.yaml', epochs=50, imgsz=255, device='cpu')
    
    model.train(data=data, epochs=2, imgsz=255, device='cpu',conf=conf_train)
    
    model_path = create_folder(model_name, 3)
    move_model_train(model_path)

    try:
            new_model = models.Model_Gen(
                usuario_id= user_id,
                model_name=model_name,
                path=model_path,
                labels=labels,
                conf = conf_train
            )
            db.add(new_model)
            db.commit()
            db.refresh(new_model)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Error creando Modelo {e}"
        )

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
        
#clase que actua como tail, lee y retorna el ultimo registro agregado al excel
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

def create_train_tmp():
    tmp_path= os.path.join(os.getcwd(),'runs\detect')

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    else:
        for item in os.listdir(tmp_path):
            item_path = os.path.join(tmp_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

def delete_folder_predic_tmp():
    tmp_path= os.path.join(os.getcwd(),'runs\detect')

    if os.path.exists(tmp_path):
        for item in os.listdir(tmp_path):
            item_path = os.path.join(tmp_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

def move_model_train(destination_path):
    source_path= os.path.join(os.getcwd(),r"runs\detect\train") 
    print(os.path.exists(source_path))
    # Mover todos los contenidos de 'source_path' a 'destination_path'
    for item in os.listdir(source_path):
        item_path = os.path.join(source_path, item)
        if os.path.isdir(item_path):
            # Si es un directorio, muévelo recursivamente
            shutil.move(item_path, destination_path)
        else:
            # Si es un archivo, muévelo directamente
            shutil.move(item_path, destination_path)

def move_annotation_tmp_for_train(source_path):
    destination_path= os.path.join(os.getcwd(),r"models\tmp") 

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    else:
        for item in os.listdir(destination_path):
            item_path = os.path.join(destination_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    print(os.path.exists(destination_path))
    # Mover todos los contenidos de 'source_path' a 'destination_path'
    for item in os.listdir(source_path):
        item_path = os.path.join(source_path, item)
        if os.path.isdir(item_path):
            # Si es un directorio, muévelo recursivamente
            shutil.copytree(item_path, os.path.join(destination_path, item))  
        else:
            # Si es un archivo, muévelo directamente
            shutil.copy(item_path, destination_path)

async def get_predict(detect_name, model_id, user_id, files,db):
    model = db.query(models.Model_Gen).filter(models.Model_Gen.id == model_id).first()
    detection_path = create_folder(detect_name, 4)
    delete_folder_predic_tmp()
    #model_best = os.path.join(model.path,  r"weights\best.pt") 
    model_best = os.path.join(os.getcwd(),r"best.pt") 
    #labels = annotation.labels
    detect_name= 'predict'
    #path = os.getcwd()
    #output_images_path = path + "/" + detect_name

    # if not os.path.exists(output_images_path):
    #     os.makedirs(output_images_path)
    
    count = 0
    model = YOLO(model_best)

    detections = []  # Lista para almacenar las detecciones en formato JSON
    detected_image_data = []
    bytes_image = io.BytesIO()
    for file in files:
        #model = YOLO('yolov8n.pt')
        #image = np.array(Image.open(file.file))

        image = cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        if image is None:
            continue
        if file.filename.split(".")[-1] not in ["jpg","jpeg", "png"]:
            continue
        
        resized = redimension(image, 640)
        # name = f"/storage/{str(uuid.uuid4())}.jpg"
        # cv2.imwrite(name, output)
        results =  model(resized, stream=True,save=True,device='cpu',conf=conf_detect)
        #cv2.imshow("result", res_plotted)
        class_data = []
        for r in results:
            boxes = r.boxes
            prueba = r.probs
            res_plotted = r.plot(probs=True)
            #cv2.imshow("result", res_plotted)
            #cv2.waitKey(1)
            

            for box in boxes:
                
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #cv2.rectangle(img,(x1, y1),(x2,y2),(255,0,255),3)
                
                w, h = x2 -x1, y2 -y1
                
                #Confianza
                conf = math.ceil((box.conf[0] * 100 )) / 100
                #Class Name
                cls = int(box.cls[0])
                myColor= (0, 255, 0)
                currentClass = r.names[cls]

                class_item = {
                "cls": currentClass,
                "conf": conf
                }
                class_data.append(class_item)

                if conf >= conf_detect:
                    cvzone.cornerRect(resized, (x1, y1, w, h),l=8,rt=5)
                    ##currenArray = np.array([x1, y1, x2, y2, conf])
                    cvzone.putTextRect(resized, f'{r.names[cls]} {conf}', 
                                    (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                    colorT=(255,255,255),colorR=myColor, offset=5)
                    
                    #cv2.rectangle(resized,(x1,y1), (x2, y2), myColor, 3)

                    detections.append({
                        "cls": currentClass,
                        "conf": conf,
                        "bounding_box": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }
                    })
            impru = cv2.imread(os.path.join("runs/detect/predict",r.path))
            # detected_image_data = {
            # "image_bytes": impru.tobytes(),
            # "name_images": r.path,
            # "clases": class_data
            # }
            image_bytes = cv2.imencode('.jpg', impru)[1].tobytes()
            detected_image_data.append({
            "image_bytes":base64.b64encode(image_bytes).decode('utf-8'),
            "name_images": r.path,
            "clases": class_data
            })

        #cv2.imwrite(detection_path + "/image" + str(count) + ".jpg", resized)        
        cv2.imwrite(os.path.join(detection_path, f"image{count}.jpg"), resized)
        impru = cv2.imread(os.path.join("runs/detect/predict",r.path))      
        cv2.imwrite(os.path.join(detection_path, f"pruimage{count}.jpg"), impru)
        count += 1
        bytes_image

        cv2.imwrite(os.path.join(detection_path, f"image{count}.jpg"), resized)
        impru = cv2.imread(os.path.join("runs/detect/predict","image0.jpg"))      
        cv2.imwrite(os.path.join(detection_path, f"pruimage{count}.jpg"), impru)
    results={"message":"This is just test message"}

    response_data = {
        "images": detected_image_data
    }

    return JSONResponse(content=response_data)
    #image_bytes = 
    # return FileResponse(os.path.join(detection_path, f"pruimage{count}.jpg"),headers=results)
    #files = [("files", ("name",   impru.tobytes(), "JPG")) for file in uploaded_file]


    #return Response(content= impru.tobytes(), media_type="image/jpeg",headers=results)  Funciona
    #return StreamingResponse(io.BytesIO(impru.tobytes()), media_type="image/jpeg")






def readImage(filename,imagespath):
    # OpenCV uses BGR channels
    img = cv2.imread(os.path.join(imagespath,filename))
    return img