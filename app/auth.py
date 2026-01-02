from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import User
import hashlib

router = APIRouter()

def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

@router.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    db: Session = SessionLocal()
    user = User(username=username, password=hash_pwd(password))
    db.add(user)
    db.commit()
    return RedirectResponse("/login", status_code=302)

@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    user = db.query(User).filter(
        User.username == username,
        User.password == hash_pwd(password)
    ).first()

    if user:
        request.session["user"] = username
        return RedirectResponse("/", status_code=302)

    return RedirectResponse("/login", status_code=302)
