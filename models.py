from sqlalchemy import Column, String, LargeBinary, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(UUID(as_uuid=True), primary_key=True)
    email = Column(String(255), unique=True, nullable=False)

class FaceIdentity(Base):
    __tablename__ = 'face_identities'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=True)
    embedding = Column(LargeBinary)
    num_images_used = Column(Integer, default=0)
    embedding_quality = Column(Integer, default=0)
    created_at = Column(String)
    updated_at = Column(String)
