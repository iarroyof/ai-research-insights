from __future__ import annotations
from neo4j import GraphDatabase
from app.config import settings

_driver = None

def neo_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo.uri,
            auth=(settings.neo.user, settings.neo.password)
        )
    return _driver

def neo_session():
    return neo_driver().session()
