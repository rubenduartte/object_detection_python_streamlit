import streamlit as st
from streamlit_image_annotation import detection
from glob import glob
from PIL import Image
import pandas as pd
from API import API
import json
from pandas import json_normalize

from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder


@st.cache_resource
def getdataset():
    return api.get_dataset()

def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)
    st.write(st.session_state)
    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )
    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    pru = edited_df[edited_df.Select]["Select"] 
    st.write(pru)
    return selected_rows.drop('Select', axis=1)

def anotation(labels):

    label_list = []
    if labels and labels.strip():
        label_list.append(labels.split(','))

    #label_list = ['coca', 'pepsi', 'facebook', 'penguin', 'framingo', 'teddy bear']
  

    # Obtenemos todos los archivos en el directorio que terminan con ".jpg"
    jpg_files = glob('model/frames/train/images/*.jpg', recursive=False)
    # Obtenemos todos los archivos en el directorio que terminan con ".png"
    png_files = glob('model/frames/train/images/*.png', recursive=False)
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
        with open("model/frames/train/images/"+ file_name, 'w') as f:
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

    st.json(st.session_state['result_dict'])


st.markdown("""
<style>
.css-czk5ss.e16jpq800
{
    visibility: hidden;
}                      
</style>
            
            """, unsafe_allow_html=True)
api = API("http://127.0.0.1:8000","asdsadsadsada")


data_json = getdataset()

df = json_normalize(data_json) 

gd = GridOptionsBuilder.from_dataframe(df.iloc[:,[3,2,0,4]]) #ordenamos el datatable
gd.configure_pagination(enabled=True)
gd.configure_default_column(editable=False, groupable=False)
gd.configure_selection(selection_mode="single", use_checkbox=True)

gd.configure_column("id","ID",width=150)
gd.configure_column("usuario_id","ID USuario",hide=True) 
gd.configure_column("labesls","Etiquetas",width=350)
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
    st.write('Why hello there')
    anotation(sel_row[0]['labesls'])







