import streamlit as st
import os
from streamlit_tags import st_tags, st_tags_sidebar
import requests
from PIL import Image
import io
from pydantic import BaseModel


def main():
    st.title("Cargar Dataset")
    dataset_name ="prueba"
    keywords = st_tags(
            label='# Etiquetas:',
            text='Presione Enter para agregar',            
            maxtags = 10,
            key='1')
        
    uploaded_file = st.file_uploader("Cargar Imagenes", type=["jpg", "jpeg", "png"],accept_multiple_files=True)
    
    if st.button("Guardar"):

        if len(uploaded_file)!=0:
            st.write("ACEPTADO")

            files = [("files", (file.name,   file.getvalue(), file.type)) for file in uploaded_file]


            URL=f"http://localhost:8000/load_image"


            if keywords is not None and len(keywords) != 0:
                keywords = ','.join(keywords)
            data  = {
                "dataset_name": dataset_name, 
                "labels": keywords
            }
            response = requests.post(URL,data=data ,files=files)
            print(response)

def save_uploaded_file(uploaded_file):
    # Crear una carpeta temporal para guardar las imágenes
    os.makedirs("temp", exist_ok=True)

    # Guardar la imagen en la carpeta temporal
    image_path = os.path.join("temp", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return image_path

def save_label(image_path, label):
    # Crear una carpeta para guardar los archivos de etiquetas
    os.makedirs("labels", exist_ok=True)

    # Guardar la etiqueta en un archivo de texto con el mismo nombre que la imagen
    label_path = os.path.join("labels", os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    with open(label_path, "w") as f:
        f.write(label)



if __name__ == "__main__":
    main()
