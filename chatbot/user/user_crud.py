# user_crud.py

from sqlalchemy.orm import Session
from models import User, UserCreate
from user.user_schema import NewUserForm

from passlib.context import CryptContext

# Password hashing context using bcrypt algorithm
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    """
    Verify if the plain password matches the hashed password.

    Args:
        plain_password (str): The plain text password input by the user.
        hashed_password (str): The hashed password stored in the database.

    Returns:
        bool: True if the password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_user(user_name: str, db: Session):
    """
    Retrieve a user from the database by username.

    Args:
        user_name (str): The username to search for.
        db (Session): The database session.

    Returns:
        User or None: The user object if found, else None.
    """
    return db.query(User).filter(User.user_name == user_name).first()

def create_user(new_user: NewUserForm, db: Session):
    """
    Create a new user in the database.

    Args:
        new_user (NewUserForm): The new user's data.
        db (Session): The database session.
    """
    user = User(
        user_name=new_user.user_name, user_pw=pwd_context.hash(new_user.password)
    )
    db.add(user)
    db.commit()

def get_user(db: Session, email: str):
    """
    Retrieve a user from the database by email.

    Args:
        db (Session): The database session.
        email (str): The email to search for.

    Returns:
        User or None: The user object if found, else None.
    """
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate):
    """
    Create a new user in the database.

    Args:
        db (Session): The database session.
        user (UserCreate): The user creation data.

    Returns:
        User: The newly created user object.
    """
    hashed_password = pwd_context.hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, email: str, password: str):
    """
    Authenticate a user by verifying their email and password.

    Args:
        db (Session): The database session.
        email (str): The user's email.
        password (str): The plain text password input by the user.

    Returns:
        User or bool: The user object if authentication is successful, else False.
    """
    user = get_user(db, email)
    if user is None:
        return False
    if not pwd_context.verify(password, user.hashed_password):
        return False
    return user
