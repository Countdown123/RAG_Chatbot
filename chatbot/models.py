from sqlalchemy import Column, Integer, VARCHAR, DateTime
from datetime import datetime

from database import Base

class User(Base):
  __tablename__ = "Users"
  
  user_id = Column(Integer, primary_key=True, autoincrement=True)
  user_name = Column(VARCHAR(10), nullable=False)
  user_pw=Column(VARCHAR(100), nullable=False)
  status=Column(VARCHAR(1), nullable=False, default='1')
  signin_date = Column(DateTime, nullable=False, default=datetime.now)