import os
import uuid
import shutil
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Set

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr
from jose import JWTError, jwt
from passlib.context import CryptContext

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    BigInteger,
    UniqueConstraint,
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

DATABASE_URL = "sqlite:///./app.db"

# Use env var in production
SECRET_KEY = os.getenv("DISCORDLITE_SECRET_KEY", "CHANGE_THIS_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_MESSAGE_LENGTH = 2000
MAX_USERNAME_LENGTH = 32

# ------------------------------------------------------------------------------
# DB Setup
# ------------------------------------------------------------------------------

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(MAX_USERNAME_LENGTH), unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="user")
    files = relationship("FileRecord", back_populates="user")


class Channel(Base):
    __tablename__ = "channels"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    is_dm = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="channel")
    files = relationship("FileRecord", back_populates="channel")
    members = relationship("ChannelMember", back_populates="channel")


class ChannelMember(Base):
    __tablename__ = "channel_members"
    __table_args__ = (
        UniqueConstraint("channel_id", "user_id", name="uq_channel_user"),
    )

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    channel = relationship("Channel", back_populates="members")
    user = relationship("User")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    channel = relationship("Channel", back_populates="messages")
    user = relationship("User", back_populates="messages")


class FileRecord(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    original_filename = Column(String(255), nullable=False)
    stored_path = Column(String(500), nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    channel = relationship("Channel", back_populates="files")
    user = relationship("User", back_populates="files")


Base.metadata.create_all(bind=engine)

# ------------------------------------------------------------------------------
# Auth utilities
# ------------------------------------------------------------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------------------

UsernameStr = constr(strip_whitespace=True, min_length=3, max_length=MAX_USERNAME_LENGTH)
PasswordStr = constr(strip_whitespace=True, min_length=6, max_length=128)
MessageContentStr = constr(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_LENGTH)


class Token(BaseModel):
    access_token: str
    token_type: str


class UserCreate(BaseModel):
    username: UsernameStr
    password: PasswordStr


class UserOut(BaseModel):
    id: int
    username: str

    class Config:
        orm_mode = True


class ChannelCreate(BaseModel):
    name: constr(strip_whitespace=True, min_length=1, max_length=100)


class ChannelOut(BaseModel):
    id: int
    name: str
    is_dm: bool

    class Config:
        orm_mode = True


class MessageCreate(BaseModel):
    content: MessageContentStr


class MessageOut(BaseModel):
    id: int
    channel_id: int
    user: UserOut
    content: str
    created_at: datetime

    class Config:
        orm_mode = True


class FileOut(BaseModel):
    id: int
    channel_id: int
    user: UserOut
    original_filename: str
    size_bytes: int
    created_at: datetime

    class Config:
        orm_mode = True


# ------------------------------------------------------------------------------
# FastAPI app & CORS
# ------------------------------------------------------------------------------

app = FastAPI(title="Discord-lite Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Auth dependencies
# ------------------------------------------------------------------------------

from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: Optional[int] = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise credentials_exception
    return user


def ensure_channel_member(db: Session, user: User, channel_id: int) -> Channel:
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    membership = (
        db.query(ChannelMember)
        .filter(ChannelMember.channel_id == channel_id, ChannelMember.user_id == user.id)
        .first()
    )
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this channel")
    return channel


# ------------------------------------------------------------------------------
# REST Endpoints
# ------------------------------------------------------------------------------

@app.post("/auth/register", response_model=UserOut)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user_in.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        username=user_in.username,
        password_hash=get_password_hash(user_in.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Automatically create a "General" channel if none exist and add user to it
    general = db.query(Channel).filter(Channel.name == "general").first()
    if not general:
        general = Channel(name="general", is_dm=False)
        db.add(general)
        db.commit()
        db.refresh(general)
    member = ChannelMember(channel_id=general.id, user_id=user.id)
    db.add(member)
    db.commit()

    return user


@app.post("/auth/login", response_model=Token)
def login(form_data: UserCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/me", response_model=UserOut)
def read_me(current_user: User = Depends(get_current_user)):
    return current_user


# Channels ---------------------------------------------------------------------

@app.get("/channels", response_model=List[ChannelOut])
def list_channels(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    channels = (
        db.query(Channel)
        .join(ChannelMember, Channel.id == ChannelMember.channel_id)
        .filter(ChannelMember.user_id == current_user.id)
        .all()
    )
    return channels


@app.post("/channels", response_model=ChannelOut)
def create_channel(
    channel_in: ChannelCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    channel = Channel(name=channel_in.name, is_dm=False)
    db.add(channel)
    db.commit()
    db.refresh(channel)

    member = ChannelMember(channel_id=channel.id, user_id=current_user.id)
    db.add(member)
    db.commit()

    return channel


@app.post("/channels/{channel_id}/join", response_model=ChannelOut)
def join_channel(
    channel_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    existing = (
        db.query(ChannelMember)
        .filter(ChannelMember.channel_id == channel_id, ChannelMember.user_id == current_user.id)
        .first()
    )
    if existing:
        return channel

    member = ChannelMember(channel_id=channel_id, user_id=current_user.id)
    db.add(member)
    db.commit()
    return channel


# Messages ---------------------------------------------------------------------

@app.get("/channels/{channel_id}/messages", response_model=List[MessageOut])
def get_messages(
    channel_id: int,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ensure_channel_member(db, current_user, channel_id)

    limit = max(1, min(limit, 200))  # clamp 1..200
    msgs = (
        db.query(Message)
        .filter(Message.channel_id == channel_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    # return newest first; front-end can reverse if desired
    return list(reversed(msgs))


@app.post("/channels/{channel_id}/messages", response_model=MessageOut)
def post_message(
    channel_id: int,
    msg_in: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ensure_channel_member(db, current_user, channel_id)

    message = Message(
        channel_id=channel_id,
        user_id=current_user.id,
        content=msg_in.content,
    )
    db.add(message)
    db.commit()
    db.refresh(message)

    # Broadcast via WebSocket manager (if any connections)
    WebSocketManager.broadcast_message(
        channel_id,
        {
            "type": "message",
            "id": message.id,
            "channel_id": channel_id,
            "user": {"id": current_user.id, "username": current_user.username},
            "content": message.content,
            "created_at": message.created_at.isoformat(),
        },
    )

    return message


# Files ------------------------------------------------------------------------

@app.post("/channels/{channel_id}/files", response_model=FileOut)
async def upload_file(
    channel_id: int,
    upload: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ensure_channel_member(db, current_user, channel_id)

    # Read file content and enforce size
    contents = await upload.read()
    size = len(contents)
    if size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large, max {MAX_FILE_SIZE_MB}MB.",
        )

    # Store file on disk with unique name
    extension = os.path.splitext(upload.filename)[1]
    unique_name = f"{uuid.uuid4().hex}{extension}"
    stored_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(stored_path, "wb") as f:
        f.write(contents)

    record = FileRecord(
        channel_id=channel_id,
        user_id=current_user.id,
        original_filename=upload.filename,
        stored_path=stored_path,
        size_bytes=size,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # Broadcast file event via WebSocket
    WebSocketManager.broadcast_message(
        channel_id,
        {
            "type": "file",
            "id": record.id,
            "channel_id": channel_id,
            "user": {"id": current_user.id, "username": current_user.username},
            "original_filename": record.original_filename,
            "size_bytes": record.size_bytes,
            "created_at": record.created_at.isoformat(),
        },
    )

    return record


@app.get("/files/{file_id}")
def download_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    file_rec = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not file_rec:
        raise HTTPException(status_code=404, detail="File not found")

    # Permission check: must be in the file's channel
    ensure_channel_member(db, current_user, file_rec.channel_id)

    if not os.path.exists(file_rec.stored_path):
        raise HTTPException(status_code=410, detail="File no longer available")

    return FileResponse(
        path=file_rec.stored_path,
        filename=file_rec.original_filename,
        media_type="application/octet-stream",
    )


# ------------------------------------------------------------------------------
# WebSocket Manager
# ------------------------------------------------------------------------------

class WebSocketManager:
    """
    Manages WebSocket connections per channel.
    """
    # channel_id -> set of WebSockets
    connections: Dict[int, Set[WebSocket]] = {}

    @classmethod
    async def connect(cls, channel_id: int, websocket: WebSocket):
        await websocket.accept()
        cls.connections.setdefault(channel_id, set()).add(websocket)

    @classmethod
    def disconnect(cls, channel_id: int, websocket: WebSocket):
        if channel_id in cls.connections:
            cls.connections[channel_id].discard(websocket)
            if not cls.connections[channel_id]:
                del cls.connections[channel_id]

    @classmethod
    def broadcast_message(cls, channel_id: int, message: dict):
        if channel_id not in cls.connections:
            return
        dead = []
        for ws in cls.connections[channel_id]:
            try:
                # send_text is async, but to keep API simple we "fire and forget"
                # You could make this async and await all in a proper loop.
                import asyncio
                asyncio.create_task(ws.send_json(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            cls.disconnect(channel_id, ws)


# ------------------------------------------------------------------------------
# WebSocket endpoint (auth required via ?token=...)
# ------------------------------------------------------------------------------

@app.websocket("/ws/channels/{channel_id}")
async def websocket_endpoint(websocket: WebSocket, channel_id: int):
    """
    Connect with:
    ws://host/ws/channels/{channel_id}?token=JWT_HERE
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Verify token and membership
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: Optional[int] = payload.get("sub")
        if user_id is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    except JWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        try:
            ensure_channel_member(db, user, channel_id)
        except HTTPException:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    finally:
        db.close()

    # Connection accepted
    await WebSocketManager.connect(channel_id, websocket)

    try:
        while True:
            data = await websocket.receive_text()
            # Simple echo: treat text from WebSocket as a message
            # In practice, you'll parse JSON for richer actions.
            if not data.strip():
                continue

            # Store message in DB then broadcast
            db = SessionLocal()
            try:
                message = Message(
                    channel_id=channel_id,
                    user_id=int(user_id),
                    content=data[:MAX_MESSAGE_LENGTH],
                )
                db.add(message)
                db.commit()
                db.refresh(message)

                WebSocketManager.broadcast_message(
                    channel_id,
                    {
                        "type": "message",
                        "id": message.id,
                        "channel_id": channel_id,
                        "user": {"id": int(user_id)},
                        "content": message.content,
                        "created_at": message.created_at.isoformat(),
                    },
                )
            finally:
                db.close()

    except WebSocketDisconnect:
        WebSocketManager.disconnect(channel_id, websocket)
