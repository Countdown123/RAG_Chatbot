# database.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DB_URL = "sqlite:///./fastapi.db"

# Create the SQLAlchemy engine
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative class definitions
Base = declarative_base()

def get_db():
    """
    Dependency that provides a database session to routes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
