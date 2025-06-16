from tests.fixtures.configurations import *  # noqa: F403
#from tests.fixtures.open_ai_service import *  # noqa: F403
from tests.fixtures.paths import * # noqa: F403
from tests.fixtures.test_data import * # noqa: F403
import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
