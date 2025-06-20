from tests.fixtures.configurations import *  # noqa: F403
from tests.fixtures.open_ai_service import *  # noqa: F403
from tests.fixtures.paths import * # noqa: F403
from tests.fixtures.test_data import * # noqa: F403
from tests.fixtures.database import * # noqa: F403
from tests.fixtures.mock_database import * # noqa: F403
from tests.fixtures.postgresql import * # noqa: F403
import pytest
import sys
import asyncio


@pytest.fixture(scope="session", autouse=True)
def configure_windows_event_loop_policy():
    """Configure Windows to use the selector event loop policy."""
    if sys.platform.startswith('win'):
        # Windows-specific event loop policy fix
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    yield


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"