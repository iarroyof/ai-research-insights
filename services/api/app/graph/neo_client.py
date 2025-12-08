from __future__ import annotations
from neo4j import GraphDatabase
from app.config import settings

_driver = None

def neo_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.user, settings.neo4j.password)
        )
    return _driver

def neo_session():
    return neo_driver().session()
