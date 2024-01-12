from db.datebase import Base
from sqlalchemy import Column, Integer, String, Boolean,DateTime,Float
from datetime import datetime
from sqlalchemy.schema import ForeignKey
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = "user"
    id = Column(Integer,primary_key=True,autoincrement=True)
    username = Column(String,unique=True)
    password = Column(String)
    nombre = Column(String)
    apellido = Column(String)
    direccion = Column(String)
    telefono = Column(Integer)
    correo = Column(String,unique=True)
    creacion = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    estado = Column(Boolean,default=False)
    dataset = relationship("Dataset",backref="User",cascade="delete,merge")
    annotation = relationship("Annotation",backref="User",cascade="delete,merge")
    model_gen = relationship("Model_Gen",backref="User",cascade="delete,merge")
    role_id = Column(Integer,ForeignKey("role.id",ondelete="CASCADE"))
    role = relationship("Role",backref="User",cascade="delete,merge")

class Dataset(Base):
    __tablename__ = "dataset"
    id = Column(Integer,primary_key=True,autoincrement=True)
    usuario_id = Column(Integer,ForeignKey("user.id",ondelete="CASCADE"))
    dataset_name = Column(String)
    path = Column(String)
    labels = Column(String)

class Model_Gen(Base):
    __tablename__ = "model_gen"
    id = Column(Integer,primary_key=True,autoincrement=True)
    usuario_id = Column(Integer,ForeignKey("user.id",ondelete="CASCADE"))
    model_name = Column(String)
    path = Column(String)
    creacion = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    labels = Column(String)
    conf = Column(Float)

class Role(Base):
    __tablename__ = "role"
    id = Column(Integer,primary_key=True,autoincrement=True)
    name = Column(String,unique=True)
    description = Column(String)
    status = Column(Boolean,default=False)

class Annotation(Base):
    __tablename__ = "annotation"
    id = Column(Integer,primary_key=True,autoincrement=True)
    usuario_id = Column(Integer,ForeignKey("user.id",ondelete="CASCADE"))
    annotation_name = Column(String)
    path = Column(String)
    labels = Column(String)