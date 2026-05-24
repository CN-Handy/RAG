import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session as _Session

from app.config import config, resolve_path

_engine_type = config["database"]["engine"]
if _engine_type == "sqlite":
    _db_url = f"sqlite:///{resolve_path(config['database']['path'])}"
else:
    _db_url = (
        f"mysql+pymysql://{config['database']['username']}:{config['database']['password']}"
        f"@{config['database']['host']}:{config['database']['port']}/rag"
    )

engine = create_engine(_db_url, connect_args={"check_same_thread": False} if _engine_type == "sqlite" else {})
Base = declarative_base()
Session = sessionmaker(bind=engine)


class KnowledgeDatabase(Base):
    __tablename__ = "knowledge_database"

    knowledge_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    category = Column(String)
    create_dt = Column(DateTime, default=datetime.datetime.now)
    update_dt = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)

    def __str__(self):
        return f"KnowledgeDatabase(id={self.knowledge_id}, title='{self.title}', category='{self.category}')"


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_document"

    document_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    category = Column(String)
    knowledge_id = Column(Integer, ForeignKey("knowledge_database.knowledge_id"), nullable=False)
    file_path = Column(String)
    file_type = Column(String)
    # pending → processing → completed / failed
    parse_status = Column(String, default="pending")
    create_dt = Column(DateTime, default=datetime.datetime.now)
    update_dt = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)

    def __str__(self):
        return f"KnowledgeDocument(id={self.document_id}, title='{self.title}', knowledge_id={self.knowledge_id})"


Base.metadata.create_all(engine)
