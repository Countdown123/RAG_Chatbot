# user_router.py

from sqlalchemy.orm import Session
from database import get_db

from datetime import datetime
from datetime import timedelta
import jwt

from fastapi import APIRouter, Depends, status, HTTPException, Response, Request
from fastapi.security import OAuth2PasswordRequestForm

from user import user_schema, user_crud

import os
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Create a new APIRouter instance with a prefix
app = APIRouter(prefix="/user")


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """
    Create a JWT access token.

    Args:
        data (dict): The data to include in the token payload.
        expires_delta (timedelta | None, optional): The expiration time delta.

    Returns:
        str: The encoded JWT token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.post(path="/signup")
async def signup(new_user: user_schema.NewUserForm, db: Session = Depends(get_db)):
    """
    Endpoint for user signup.

    Args:
        new_user (user_schema.NewUserForm): The new user's data.
        db (Session): Database session dependency.

    Returns:
        HTTPException: HTTP response indicating the result.
    """
    # 회원 존재 여부 확인
    user = user_crud.get_user(new_user.user_name, db)

    if user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="User already exists"
        )

    # 회원 가입
    user_crud.create_user(new_user, db)

    return HTTPException(status_code=status.HTTP_200_OK, detail="Signup successful")


@app.post(path="/login")
async def login(
    response: Response,
    login_form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Endpoint for user login.

    Args:
        response (Response): Response object to set cookies.
        login_form (OAuth2PasswordRequestForm): Form data for login.
        db (Session): Database session dependency.

    Returns:
        user_schema.Token: Access token information.
    """
    # 회원 존재 여부 확인
    user = user_crud.get_user(login_form.username, db)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user or password"
        )

    # 로그인
    res = user_crud.verify_password(login_form.password, user.user_pw)

    # 토큰 생성
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_name}, expires_delta=access_token_expires
    )

    # 쿠키에 저장
    response.set_cookie(
        key="access_token",
        value=access_token,
        expires=access_token_expires,
        httponly=True,
    )

    if not res:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user or password"
        )

    return user_schema.Token(access_token=access_token, token_type="bearer")


@app.get(path="/logout")
async def logout(response: Response, request: Request):
    """
    Endpoint for user logout.

    Args:
        response (Response): Response object to modify cookies.
        request (Request): Request object to access cookies.

    Returns:
        HTTPException: HTTP response indicating the result.
    """
    access_token = request.cookies.get("access_token")

    # 쿠키 삭제
    response.delete_cookie(key="access_token")

    return HTTPException(status_code=status.HTTP_200_OK, detail="Logout successful")
