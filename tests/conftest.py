import pytest

from niche_miner import db
from niche_miner.settings import Settings


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def settings():
    return Settings()
