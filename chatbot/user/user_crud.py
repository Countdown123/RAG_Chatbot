from sqlalchemy.orm import Session
from models import User, UserCreate
from user.user_schema import NewUserForm

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_user(user_name: str, db: Session):  
    return db.query(User).filter(User.user_name == user_name).first()


def create_user(new_user: NewUserForm, db: Session):
    user = User(
        user_name =new_user.user_name,
        user_pw=pwd_context.hash(new_user.password)
    )
    db.add(user)
    db.commit()

def get_user(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, user: UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, email: str, password: str):
    user = get_user(db, email)
    if user is None:
        return False
    if not pwd_context.verify(password, user.hashed_password):
        return False
    return user