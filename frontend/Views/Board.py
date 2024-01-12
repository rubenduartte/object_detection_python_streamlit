import streamlit as st
from streamlit_option_menu import option_menu
from API import API
import os
import io
from streamlit_tags import st_tags
import time
from streamlit_autorefresh import st_autorefresh
from glob import glob
from streamlit_image_annotation import detection
from PIL import Image
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from pandas import json_normalize
import shutil
import aiohttp
import json
import asyncio
import base64
import pandas as pd






class Board:
    def __init__(self, username: str, api: API):

        def load_dataset():
                st.title("Cargar Dataset")

                #Se agrupa el codigo para poder limpiar
                holder = st.empty()
                with holder:
                    with st.form("my-form", clear_on_submit=True):
                        dataset_name = st.text_input('Nombre')
                        keywords = st_tags(
                                label='Etiquetas:',
                                text='Presione Enter para agregar',
                                maxtags = 10,
                                key='labelsinput')

                        uploaded_file = st.file_uploader("Cargar Imagenes", type=["jpg", "jpeg", "png"],accept_multiple_files=True)
                        submitted = st.form_submit_button("Guardar!")

                if submitted and uploaded_file is not None:

                    if len(uploaded_file)!=0:
                        user = api.get_user_from_username(username)
                        user_id = user['id']
                        files = [("files", (file.name,   file.getvalue(), file.type)) for file in uploaded_file]
                        if keywords is not None and len(keywords) != 0:
                            keywords = ','.join(keywords)
                        data  = {
                            "dataset_name": dataset_name,
                            "labels": keywords,
                            "user_id": user_id
                        }
                        response = api.save_dataset(data,files)
                        if response:
                            st.success("Guardado Exitoso")
                            time.sleep(3)

                            holder.empty()  # Limpia los componentes de entrada
                            keywords = []  # Limpia las etiquetas ingresadas
                            uploaded_file = []  # Limpia los archivos cargados
                            del st.session_state['labelsinput']
                            st_autorefresh(interval=1, limit=2, key="refresh-count") # Recarga la pagina
                        else:
                            st.error("No se pudo Guardar el dataset")


        #@st.cache_resource
        def getdataset():
            return api.get_dataset()

        #@st.cache_resource
        def getannotation():
            return api.get_annotation()

        #@st.cache_resource
        def get_models():
            return api.get_models()

        def create_dataset_tmp(path_dataset):
            # Obtener el año y mes actual

            tmp_path= os.path.join(os.getcwd(),'tmp')

            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)
            else:
                for item in os.listdir(tmp_path):
                    item_path = os.path.join(tmp_path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)

            for item in os.listdir(path_dataset):
                src_item_path = os.path.join(path_dataset, item)
                dst_item_path = os.path.join(tmp_path, item)
                if os.path.isdir(src_item_path):
                    shutil.copytree(src_item_path, dst_item_path)
                else:
                    shutil.copy(src_item_path, dst_item_path)

        def check_bboxes(json_data):
            for key, value in json_data.items():
                if isinstance(value, dict):
                    bboxes = value.get("bboxes")
                    if not bboxes:
                        return False
                    if not check_bboxes(value):
                        return False
            return True

        def load_files_from_directory(directory):
            files = []
            print("entro")
            for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        with open(file_path, 'rb') as file:
                            file_content = file.read()
                            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                                file_type = 'image/jpeg' if filename.lower().endswith('.jpg') else 'image/png'
                            elif filename.lower().endswith('.txt'):
                                file_type = 'text/plain'
                            else:
                                continue  # Ignore unsupported file types
                            files.append(("files", (filename, io.BytesIO(file_content), file_type)))

            return files



        def anotation(labels):

            label_list = labels.split(',')
            annotation_name = 'prueba'
            # if labels and labels.strip():
            #     label_list.append(labels.split(','))

            #label_list = ['coca', 'pepsi', 'facebook', 'penguin', 'framingo', 'teddy bear']


            # Obtenemos todos los archivos en el directorio que terminan con ".jpg"
            jpg_files = glob('tmp/*.jpg', recursive=False)
            # Obtenemos todos los archivos en el directorio que terminan con ".png"
            png_files = glob('tmp/*.png', recursive=False)
            image_path_list = jpg_files + png_files
            if 'etiqueta' not in st.session_state:
                st.session_state['etiqueta'] = label_list

            if 'result_dict' not in st.session_state:
                result_dict = {}
                for img in image_path_list:
                #    result_dict[img] = {'bboxes': [[0,0,100,100],[10,20,50,150]],'labels':[0,2]}
                    result_dict[img] = {'bboxes': [],'labels':[]}
                st.session_state['result_dict'] = result_dict.copy()

            num_page = st.slider('page', 0, len(image_path_list)-1, 0, key='slider')
            target_image_path = image_path_list[num_page]
            #st.write(st.session_state['result_dict'][target_image_path]['bboxes'])

            new_labels = detection(image_path=target_image_path,
                                bboxes=st.session_state['result_dict'][target_image_path]['bboxes'],
                                labels=st.session_state['result_dict'][target_image_path]['labels'],
                                label_list=label_list, key=target_image_path)
            if new_labels is not None:
                st.session_state['result_dict'][target_image_path]['bboxes'] = [v['bbox'] for v in new_labels]
                st.session_state['result_dict'][target_image_path]['labels'] = [v['label_id'] for v in new_labels]

                file_name = target_image_path.split('\\')[-1].split('.')[0] + '.txt'
                image = Image.open(target_image_path)
                ancho, altura = image.size
                with open("tmp/"+ file_name, 'w') as f:
                    for v in new_labels:
                        bbox = v['bbox']
                        label_id = v['label_id']

                        #(((x+w)+x)/2)ANCHO-Image,(((y+h)+y)/2)ALTO-Image, w / ANCHO-Image, h / ALTO-Image
                        x = round(((((bbox[0]+bbox[2])+bbox[0])/2)/ancho), 6)
                        y = round(((((bbox[1]+bbox[3])+bbox[1])/2)/altura), 6)
                        w = round((bbox[2]/ancho), 6)
                        h = round((bbox[3]/altura), 6)

                        line = f"{label_id} {x} {y} {w} {h}\n"

                        f.write(line)
                    f.close()
            if st.button('Guardar'):
                result = check_bboxes(st.session_state['result_dict'])
                if result:
                        user = api.get_user_from_username(username)
                        user_id = user['id']

                        files = load_files_from_directory('tmp/')
                        #[("files", (file.name,   file.getvalue(), file.type)) for file in uploaded_file]
                        #if keywords is not None and len(label_list) != 0:
                        #    keywords = ','.join(keywords)
                        data  = {
                            "annotation_name": annotation_name,
                            "labels": labels,
                            "user_id": user_id
                        }
                        response = api.save_annotation(data,files)
                        if response:
                            st.success('Anotaciones Guardadas')
                            time.sleep(3)
                            st.session_state.onclick_menu = True
                            st_autorefresh(interval=1, limit=2, key="refresh-count") # Recarga la pagina
                            #holder.empty()  # Limpia los componentes de entrada
                            #keywords = []  # Limpia las etiquetas ingresadas
                            #uploaded_file = []  # Limpia los archivos cargados
                            #del st.session_state['labelsinput']
                            #st_autorefresh(interval=1, limit=2, key="refresh-count") # Recarga la pagina
                        else:
                            st.error("No se pudo Guardar las Anotaciones")
                else:
                    st.info('No todas las imagenes fueron anotadas.', icon="ℹ️")

            st.json(st.session_state['result_dict'])

        #Traelos las imagenes para crear las Anotaciones
        def dataset_for_annotation():

            data_json = getdataset()

            df = json_normalize(data_json)
            df = df.reindex(columns=['id', 'dataset_name', 'labels', 'path']) #ordenamos el datatable
            gd = GridOptionsBuilder.from_dataframe(df)
            gd.configure_pagination(enabled=True)
            gd.configure_default_column(editable=False, groupable=False)
            gd.configure_selection(selection_mode="single", use_checkbox=True)

            gd.configure_column("id","ID",width=150)
            gd.configure_column("usuario_id","ID USuario",hide=True)
            gd.configure_column("labels","Etiquetas",width=350)
            gd.configure_column("dataset_name","Nombre ")
            gd.configure_column("path","Ubicacion",width=600)
            gd.configure_pagination(enabled=True,paginationAutoPageSize=False,paginationPageSize=20)

            gd.configure_columns(["usuario_id"])
            gridoptions = gd.build()

            grid_table = AgGrid(
                df,width='100%', key = 'grid_dataset',
                gridOptions=gridoptions,fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED,theme='streamlit'
            )

            sel_row = grid_table["selected_rows"]

            # # save the row indexes of the selected rows in the session state
            # pre_selected_rows = []
            # for selected_row in selected_rows:
            #         pre_selected_rows.append(selected_row['_selectedRowNodeInfo']['nodeRowIndex'])
            # st.session_state.pre_selected_rows = pre_selected_rows


            #st.dataframe (df, column_order=("id", "dataset_name","labesls"))

            # selection = dataframe_with_selections(df)
            # st.write("Your selection:")


            if st.button('Cargar') and len(sel_row) == 1:
                st.session_state.data_for_annotation = True
                st.session_state.onclick_menu = False
                st.session_state.path_dataset = sel_row[0]['path']
                st.session_state.labels_dataset = sel_row[0]['labels']
                create_dataset_tmp(sel_row[0]['path'])
                st_autorefresh(interval=1, limit=2, key="refresh-count") # Recarga la pagina
                return st.session_state.data_for_annotation
            return False


        def train_model_menu():

            model_name =  "prueba_name"
            data_json = getannotation()

            df = json_normalize(data_json)
            #df = df.reindex(columns=['id', 'annotation_name', 'labels', 'path']) #ordenamos el datatable
            df = df.reindex(columns=['id', 'labels', 'path']) #ordenamos el datatable
            gd = GridOptionsBuilder.from_dataframe(df)
            gd.configure_pagination(enabled=True)
            gd.configure_default_column(editable=False, groupable=False)
            gd.configure_selection(selection_mode="single", use_checkbox=True)

            gd.configure_column("id","ID",width=150)
            gd.configure_column("usuario_id","ID USuario",hide=True)
            gd.configure_column("labels","Etiquetas",width=350)
            #gd.configure_column("annotation_name","Nombre ")
            gd.configure_column("path","Ubicacion",width=600)
            gd.configure_pagination(enabled=True,paginationAutoPageSize=False,paginationPageSize=20)

            gridoptions = gd.build()

            grid_table = AgGrid(
                df,width='100%', key = 'grid_annotation_train',
                gridOptions=gridoptions,fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED,theme='streamlit'
            )

            sel_row = grid_table["selected_rows"]

            #status = st.empty()
            #connect = st.checkbox("Connect to WS Server")

            # progress_text = "Operation in progress. Please wait."

            # my_bar = st.progress(0, text=progress_text)

            if st.button('Entrenar') and len(sel_row) == 1:
                progress_text = "Operacion en proceso. Por favor espere."

                my_bar = st.progress(0, text=progress_text) 

                user = api.get_user_from_username(username)
                user_id = user['id']
                    #st.session_state.path_annotation = sel_row[0]['path']
                    #st.session_state.labels_annotation = sel_row[0]['labels']
                    #create_dataset_tmp(sel_row[0]['path'])
                    #st_autorefresh(interval=1, limit=2, key="refresh-count") # Recarga la pagina
                
                api.run_training(model_name, sel_row[0]['id'],user_id)
                
                asyncio.run(
                    #api.consumer_airquality(status,my_bar)
                    api.consumer_airquality(my_bar)
                )

                st.success("ENTRENAMIENTO TERMINADO")
                time.sleep(3)
                st_autorefresh(interval=1, limit=2, key="refresh-count") # Recarga la pagina

        def annotation_menu():

            if  "data_for_annotation" not in st.session_state:
                st.session_state.data_for_annotation = False

            if st.session_state.onclick_menu:
                if  "etiqueta" in st.session_state:
                    del st.session_state.etiqueta
                if  "result_dict" in st.session_state:
                    del st.session_state.result_dict
                dataset_for_annotation()
            elif st.session_state.data_for_annotation:
                anotation(st.session_state.labels_dataset)



        def on_change_menu(key):

            selection = st.session_state[key]
            if selection == 'Crear Anotaciones':
                #if  "onclick_menu" not in st.session_state:
                st.session_state.onclick_menu = True

        def detect_logo():
                
            predic_name =  "prueba_name"
            data_json = get_models()

            df = json_normalize(data_json)
            df = df.reindex(columns=['id', 'model_name', 'labels', 'creacion']) #ordenamos el datatable
            gd = GridOptionsBuilder.from_dataframe(df)
            gd.configure_pagination(enabled=True)
            gd.configure_default_column(editable=False, groupable=False)
            gd.configure_selection(selection_mode="single", use_checkbox=True)

            gd.configure_column("id","ID",width=150)
            gd.configure_column("model_name","Modelo",hide=True)
            gd.configure_column("labels","Etiquetas",width=350)
            gd.configure_column("creacion","Fecha Creacion ")
            gd.configure_pagination(enabled=True,paginationAutoPageSize=False,paginationPageSize=20)

            gridoptions = gd.build()

            grid_table = AgGrid(
                df,width='100%', key = 'grid_model_selector',
                gridOptions=gridoptions,fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED,theme='streamlit'
            )

            sel_row = grid_table["selected_rows"]

            #st.title("Cargar Imagenes a detectar")

            uploaded_file = st.file_uploader("Cargar Imagenes", type=["jpg", "jpeg", "png"],accept_multiple_files=True)

            if st.button('Detectar') and len(sel_row) == 1 and uploaded_file is not None and len(uploaded_file)!=0:
        
                    user = api.get_user_from_username(username)
                    user_id = user['id']
                    files = [("files", (file.name,   file.getvalue(), file.type)) for file in uploaded_file]
                    
                    # if keywords is not None and len(keywords) != 0:
                    #     keywords = ','.join(keywords)
                    data  = {
                        "detect_name": predic_name,
                        "user_id": user_id,
                        "model_id": sel_row[0]['id']
                    }

                    num_columns = 2  # Puedes ajustar el número de columnas según tus preferencias
                    columns = st.columns(num_columns)

                    response =  asyncio.run(api.detect_logo(data,files))
                    #st.json(response)
                    all_classes = []
                    all_confidences = []
                    for image_data in response['images']:

                        image_bytes = base64.b64decode(image_data['image_bytes'])
                        columns[0].image(image_bytes, caption=image_data['name_images'])

                        #st.image(image_bytes,caption=image_data['name_images'])

                        # Obtener todas las clases y sus valores de confianza
                        clases = {}
                        
                        for clase in image_data['clases']:
                            cls = clase['cls']
                            conf = clase['conf']
                            all_classes.append(cls)
                            all_confidences.append(conf)
                            if cls not in clases:
                                clases[cls] = []
                            clases[cls].append(conf)
                        
                        columns[1].header("Clases Detectadas:")
                        # Mostrar las clases y sus valores de confianza
                        #st.header("Clases Detectadas:")
                        for cls, confs in clases.items():
                            columns[1].subheader(f"Label: {cls}")
                            for conf in confs:
                                columns[1].text(f"Conf: {conf:.2%}")
                   
                    #df = pd.DataFrame(clases_frame)
                    df = pd.DataFrame({'cls': all_classes, 'conf': all_confidences})


                    # Estadísticas Descriptivas
                    promedio_confianza = df['conf'].mean()

                    # Distribución de Confianza
                    st.header('Estadísticas Descriptivas:')
                    st.write(f'Promedio de Confianza: {promedio_confianza:.2%}')

                    # Histograma de Confianza
                    st.header('Distribución de Confianza:')
                    st.subheader('Histograma de Confianza')
                    st.line_chart(df['conf'])

                    # Análisis por Clase
                    st.header('Análisis por Clase:')
                    st.subheader('Conteo por Clase')
                    clase_count = df['cls'].value_counts()
                    st.bar_chart(clase_count)

                    # Promedio de Confianza por Clase
                    st.subheader('Promedio de Confianza por Clase')
                    confianza_por_clase = df.groupby('cls')['conf'].mean()
                    st.bar_chart(confianza_por_clase)

        def manager_user():
            st.title("Crear Usuario")
            
            holder = st.empty()
            with holder:
                with st.form("user-form", clear_on_submit=True):
                    nombre = st.text_input("Nombre:")
                    correo = st.text_input("Correo:")
                    direccion = st.text_input("Direccion:")
                    telefono = st.text_input("Telefono:")
                    username = st.text_input("Usuario:")
                    password = st.text_input("Contraseña:", type="password")
                    rol = st.radio("Seleccione el Permiso:",
                        ["Administrador", "Consulta"])
                    estado = st.checkbox('Activo')
                    submitted = st.form_submit_button("Guardar!")
            
                if submitted :    
                    st.write("ENTRO")

        def welcome():

            st.markdown(
                """
                Este sistema es una aplicacion contruida especificamente para proyectos de aprendizaje automatico 
                y vision computaci0nal.
                ### Que quieres hacer?
                - Generar tu propio dataset
                - Crear tus anotaciones
                - Entrenar tus modelos
                - Detectar tus logos.

            """
            )


        st.title('Detección de logotipos y reconocimiento de marcas 👋')
        st.divider()

        with st.sidebar:
            selected = option_menu("Menu Principal", ["Bienvenida","Gestionar Usuario", 'Entrenar Modelo', 'Agregar Dataset','Crear Anotaciones','Buscar Logo'],
                    icons=['people-fill', 'watch','database','pencil-square','search'],on_change=on_change_menu, key='menu_5', menu_icon="cast", default_index=0)

        if selected == 'Agregar Dataset':
            load_dataset()

        if selected == 'Crear Anotaciones':
            annotation_menu()

        if selected == 'Entrenar Modelo':
            train_model_menu()

        if selected == 'Buscar Logo':
            detect_logo()
        
        if selected == 'Gestionar Usuario':
            manager_user()
        
        if selected == 'Bienvenida':
            welcome()