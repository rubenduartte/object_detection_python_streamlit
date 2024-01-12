from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class Dataset(BaseModel):
    id: str
    dataset_name: str
    path: str
    labels: str

class Model_Gen(BaseModel):
    id: str
    model_name: str
    creacion:datetime= datetime.now()
    path: str
    labels: str
    conf: float

class User(BaseModel):
    username:str
    password: str
    nombre:str
    apellido:str
    direccion:Optional[str]
    telefono:int
    correo: str
    role_id: int
    creacion:datetime= datetime.now()

class ShowUser(BaseModel):
    id:str
    username:str
    nombre:str
    correo: str
    role_id: int
    class Config():
        orm_mode = True

class UpdateUser(BaseModel):
    username:str = None
    password: str = None
    nombre:str = None
    apellido:str = None
    direccion:str = None
    telefono:int = None
    correo: str = None

class Login(BaseModel):
    username:str
    password:str

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None

class Role(BaseModel):
    name:str
    description: str
    status:str

class Annotation(BaseModel):
    id: str
    annotation_name: str
    path: str
    labels: str
