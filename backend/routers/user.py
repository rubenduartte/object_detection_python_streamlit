from fastapi import APIRouter, Depends,status
from backend.routers.schemas import User,ShowUser,UpdateUser
from db.datebase import get_db
from sqlalchemy.orm import Session
from typing import List
from backend.repository import user
from backend.routers.oauth import get_current_user
from typing import Annotated


router = APIRouter(
    prefix="/user",
    tags=["Users"]
)
 
@router.get('/is_logged_in')
def is_logged_in(current_user: User= Depends(get_current_user)):
    if current_user:
        print("entro")
    return {"message": "Token is valid"}

@router.get('/',response_model=List[ShowUser],status_code=status.HTTP_200_OK)
def obtener_usuarios( db:Session = Depends(get_db),current_user: User= Depends(get_current_user)):
    data = user.obtener_usuarios(db)

    return data

@router.get('/{user_id}',response_model=ShowUser,status_code=status.HTTP_200_OK)
def obtener_usuario(user_id:int, db:Session = Depends(get_db)):
    usuario = user.obtener_usuario(user_id,db)
    
    return usuario

@router.post('/')
def crear_usuario(usuario:User,db:Session = Depends(get_db),status_code=status.HTTP_201_CREATED):
    user.crear_usuario(usuario,db)
    return{"respuesta": "usuario creado satisfactoriamente"}

@router.delete('/')
def eliminar_usuario(user_id:int, db:Session = Depends(get_db)):
    res =  user.eliminar_usuario(user_id,db)
    return res

@router.patch('/{user_id}')
def actualizar_user(user_id:int,updateUser:UpdateUser, db:Session = Depends(get_db)):
    res = user.actualizar_user(user_id, updateUser,db )
    return res

@router.get('/get_user_from_username/{username}',response_model=ShowUser,status_code=status.HTTP_200_OK)
def obtener_user_from_username(username:str, db:Session = Depends(get_db)):
    usuario = user.obtener_user_from_username(username,db)
    
    return usuario