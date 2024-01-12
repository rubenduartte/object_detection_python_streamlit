from sqlalchemy.orm import Session
from db import models
from fastapi import HTTPException,status
from backend.hashing import Hash

def crear_usuario(usuario, db:Session):
    try:
        
        usuario = usuario.dict()
        new_user = models.User(
            username=usuario["username"],
            password= Hash.hash_password(usuario["password"]),
            nombre=usuario["nombre"],
            apellido=usuario["apellido"],
            direccion=usuario["direccion"],
            telefono=usuario["telefono"],
            correo= usuario["correo"]
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Error creando usuario {e}"
        )
    

def obtener_usuario(user_id:int, db:Session):
    usuario = db.query(models.User).filter(models.User.id == user_id).first()

    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe el usuario con el id {user_id}"
        )
    
    return usuario

def eliminar_usuario(user_id:int, db:Session):
    usuario = db.query(models.User).filter(models.User.id == user_id)

    if not usuario.first():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe el usuario con el id {user_id} por lo tanto no se elimina"
        )
    usuario.delete(synchronize_session=False)
    db.commit()    
    return {"respuesta":"Usuario Eliminado!!"}

def obtener_usuarios( db:Session ):
    data = db.query(models.User).all()

    return data

def actualizar_user(user_id:int,updateUser, db:Session):
    usuario = db.query(models.User).filter(models.User.id == user_id)
    
    if not usuario.first():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe el usuario con el id {user_id} por lo tanto no se actualiza"
        )
    
    usuario.update(updateUser.dict(exclude_unset=True))
    db.commit()    
    return {"respuesta":"Usuario Actualizado Correctamente!!"}

def obtener_user_from_username(username:str, db:Session):
    usuario = db.query(models.User).filter(models.User.username == username).first()

    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe el usuario con el id {username}"
        )
    
    return usuario
