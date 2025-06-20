import pytest
from unittest.mock import MagicMock, AsyncMock
from ol_ai_services.tools.database.postgresql_tool_bclearer import (
    PostgreSQLConnectionConfig,
    PostgreSQLToolBclearer,
    PostgreSQLSchemaInspectionToolBclearer
)

@pytest.fixture
def postgresql_connection_config():
    """Returns a PostgreSQL connection configuration object."""
    return PostgreSQLConnectionConfig(
        host="test_host",
        port=5432,
        database="test_db",
        user="test_user",
        password="test_password"
    )

@pytest.fixture
def mock_postgresql_tool():
    """Returns a mock PostgreSQL tool."""
    mock_tool = AsyncMock(spec=PostgreSQLToolBclearer)
    
    # Configure the mock tool's run method to return test data
    mock_tool.run.return_value = MagicMock(
        success=True,
        results=[
            {"id": 1, "name": "test_row_1", "value": "value_1"},
            {"id": 2, "name": "test_row_2", "value": "value_2"}
        ],
        row_count=2,
        column_names=["id", "name", "value"]
    )
    
    return mock_tool

@pytest.fixture
def mock_postgresql_schema_tool():
    """Returns a mock PostgreSQL schema inspection tool."""
    mock_schema_tool = AsyncMock(spec=PostgreSQLSchemaInspectionToolBclearer)
    
    # Configure the mock tool's run method to return table list data
    mock_schema_tool.run.return_value = MagicMock(
        success=True,
        tables=["users", "products", "orders"],
        columns=None
    )
    
    return mock_schema_tool

@pytest.fixture
def mock_postgresql_column_tool():
    """Returns a mock PostgreSQL schema inspection tool for column details."""
    mock_column_tool = AsyncMock(spec=PostgreSQLSchemaInspectionToolBclearer)
    
    # Configure the mock tool's run method to return column data
    mock_column_tool.run.return_value = MagicMock(
        success=True,
        tables=None,
        columns=[
            {
                "column_name": "id",
                "data_type": "integer", 
                "character_maximum_length": None, 
                "column_default": "nextval('users_id_seq'::regclass)", 
                "is_nullable": "NO"
            },
            {
                "column_name": "name",
                "data_type": "character varying", 
                "character_maximum_length": 255, 
                "column_default": None, 
                "is_nullable": "NO"
            },
            {
                "column_name": "email",
                "data_type": "character varying", 
                "character_maximum_length": 255, 
                "column_default": None, 
                "is_nullable": "NO"
            },
            {
                "column_name": "created_at",
                "data_type": "timestamp", 
                "character_maximum_length": None, 
                "column_default": "CURRENT_TIMESTAMP", 
                "is_nullable": "NO"
            }
        ]
    )
    
    return mock_column_tool

@pytest.fixture
def postgresql_tool_mock(postgresql_connection_config, monkeypatch):
    """Returns a PostgreSQL tool with mocked connection."""
    # Create the tool
    tool = PostgreSQLToolBclearer(postgresql_connection_config)
    
    # Mock the _connect and _disconnect methods
    async def mock_connect(self):
        self._facade = MagicMock()
    
    async def mock_disconnect(self):
        self._facade = None
        
    monkeypatch.setattr(PostgreSQLToolBclearer, "_connect", mock_connect)
    monkeypatch.setattr(PostgreSQLToolBclearer, "_disconnect", mock_disconnect)
    
    return tool

@pytest.fixture
def postgresql_schema_tool_mock(postgresql_connection_config, monkeypatch, postgresql_tool_mock):
    """Returns a PostgreSQL schema inspection tool with mocked connection."""
    # Create the tool
    tool = PostgreSQLSchemaInspectionToolBclearer(postgresql_connection_config)
    
    # Replace the internal postgresql_tool with our mock
    tool.postgresql_tool = postgresql_tool_mock
    
    return tool