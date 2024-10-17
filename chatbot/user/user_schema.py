# user_schema.py

from pydantic import BaseModel, field_validator
from fastapi import HTTPException

class NewUserForm(BaseModel):
    """
    Pydantic model for new user registration form.
    """
    user_name: str
    password: str

    @field_validator("user_name", "password")
    def check_empty(cls, v):
        """
        Validator to check if the fields are empty or contain only whitespace.
        """
        if not v or v.isspace():
            raise HTTPException(status_code=422, detail="필수 항목을 입력해주세요.")
        return v

    @field_validator("password")
    def validate_password(cls, v):
        """
        Validator to ensure the password meets the required criteria.
        """
        if len(v) < 8:
            raise HTTPException(
                status_code=422,
                detail="비밀번호는 8자리 이상 영문과 숫자를 포함하여 작성해 주세요.",
            )

        if not any(char.isdigit() for char in v):
            raise HTTPException(
                status_code=422,
                detail="비밀번호는 8자리 이상 영문과 숫자를 포함하여 작성해 주세요.",
            )

        if not any(char.isalpha() for char in v):
            raise HTTPException(
                status_code=422,
                detail="비밀번호는 8자리 이상 영문과 숫자를 포함하여 작성해 주세요.",
            )

        return v

class Token(BaseModel):
    """
    Pydantic model for authentication token.
    """
    access_token: str
    token_type: str
