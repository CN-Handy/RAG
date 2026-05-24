import pytest
from app.db.models import KnowledgeDatabase, Session


@pytest.fixture
def session():
    with Session() as s:
        yield s
        s.rollback()


def test_insert_knowledge_database(session):
    record = KnowledgeDatabase(title="test", category="category")
    session.add(record)
    session.commit()

    result = session.query(KnowledgeDatabase).filter_by(title="test").first()
    assert result is not None
    assert result.title == "test"
    assert result.category == "category"


def test_query_knowledge_database(session):
    session.add(KnowledgeDatabase(title="test_query", category="cat"))
    session.commit()

    records = session.query(KnowledgeDatabase).filter_by(title="test_query").all()
    assert len(records) > 0
    assert records[0].title == "test_query"


def test_delete_knowledge_database(session):
    record = KnowledgeDatabase(title="test_delete", category="cat")
    session.add(record)
    session.commit()

    to_delete = session.query(KnowledgeDatabase).filter_by(title="test_delete").all()
    for r in to_delete:
        session.delete(r)
    session.commit()

    remaining = session.query(KnowledgeDatabase).filter_by(title="test_delete").all()
    assert len(remaining) == 0
