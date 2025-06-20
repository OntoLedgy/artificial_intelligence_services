import pytest
import pandas as pd
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import sys

# Create plain mocks instead of trying to patch specific imports
mock_postgresql_facade_class = MagicMock(name="PostgresqlFacade")
mock_relational_db_clients = MagicMock(name="RelationalDatabaseClients")
mock_relational_db_clients.PSYCOPG2 = "PSYCOPG2"

# Define a function to override the imports when they're attempted
def mock_import(name, *args, **kwargs):
    if name == 'bclearer_interop_services.relational_database_services.postgresql.PostgresqlFacade':
        return mock_postgresql_facade_class
    elif name == 'bclearer_interop_services.relational_database_services.source.code.object_models':
        return mock_relational_db_clients
    return original_import(name, *args, **kwargs)

# Save the original import function
original_import = __import__

# Override the import function
sys.meta_path.insert(0, type('MockImporter', (), {
    'find_spec': lambda self, fullname, path, target=None: 
        None if fullname not in [
            'bclearer_interop_services.relational_database_services.postgresql.PostgresqlFacade',
            'bclearer_interop_services.relational_database_services.source.code.object_models.RelationalDatabaseClients'
        ] else MagicMock()
}))

# Force BCLEARER_AVAILABLE to be True in our module
import os
os.environ['BCLEARER_AVAILABLE'] = 'True'

# Now import our module
# Use a try/except block to handle potential import errors
try:
    from ol_ai_services.tools.database.postgresql_tool_bclearer import (
        PostgreSQLConnectionConfig,
        PostgreSQLQueryInput,
        PostgreSQLSchemaInspectionInput,
        PostgreSQLToolBclearer,
        PostgreSQLSchemaInspectionToolBclearer
    )
    BCLEARER_AVAILABLE = True
except ImportError as e:
    # Handle import error more gracefully
    print(f"Import error: {e}")
    pytestmark = pytest.mark.skip(reason="bclearer module could not be imported")
    BCLEARER_AVAILABLE = False

# Only run tests if we could import the module
if BCLEARER_AVAILABLE:
    @pytest.fixture
    def mock_postgresql_facade():
        """Returns a mock PostgresqlFacade."""
        facade = MagicMock()
        
        # Mock for SELECT queries
        users_df = pd.DataFrame({
            'id': [1, 2, 3],
            'username': ['user1', 'user2', 'user3'],
            'email': ['user1@example.com', 'user2@example.com', 'user3@example.com']
        })
        facade.execute_query_to_dataframe.return_value = users_df
        
        # Mock for non-SELECT queries
        facade.execute_non_query.return_value = 3  # Number of affected rows
        
        return facade

    @pytest.fixture
    def bclearer_postgresql_tool(monkeypatch, mock_postgresql_facade):
        """Returns a PostgreSQL tool with mocked bclearer facade."""
        # Set a flag to bypass the actual import
        monkeypatch.setattr(
            "ol_ai_services.tools.database.postgresql_tool_bclearer.BCLEARER_AVAILABLE", 
            True
        )
        
        # Create connection config
        connection_config = PostgreSQLConnectionConfig(
            host="test_host",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_password"
        )
        
        # Create the tool
        tool = PostgreSQLToolBclearer(connection_config)
        
        # Simply set the facade directly instead of using _connect
        tool._facade = mock_postgresql_facade
        
        return tool

    @pytest.fixture
    def bclearer_schema_tool(bclearer_postgresql_tool, monkeypatch):
        """Returns a PostgreSQL schema inspection tool."""
        # Create the schema tool
        tool = PostgreSQLSchemaInspectionToolBclearer(bclearer_postgresql_tool.connection_config)
        
        # Set the mocked PostgreSQL tool
        tool.postgresql_tool = bclearer_postgresql_tool
        
        return tool

    class TestPostgreSQLToolBclearer:
        """Tests for the PostgreSQL tool using bclearer implementation."""
        
        @pytest.mark.anyio
        async def test_postgresql_select_query_execution(self, bclearer_postgresql_tool, mock_postgresql_facade):
            """Test executing a SELECT query."""
            # Create query input
            query_input = PostgreSQLQueryInput(
                query="SELECT id, username, email FROM users",
                params=None
            )
            
            # Execute the query
            result = await bclearer_postgresql_tool._run(query_input)
            
            # Verify the result
            assert result.success == True
            assert result.row_count == 3
            assert len(result.results) == 3
            assert result.results[0]["id"] == 1
            assert result.results[0]["username"] == "user1"
            assert result.results[0]["email"] == "user1@example.com"
            assert "id" in result.column_names
            assert "username" in result.column_names
            assert "email" in result.column_names
            
            # Verify the correct method was called on the facade
            mock_postgresql_facade.execute_query_to_dataframe.assert_called_once()
        
        @pytest.mark.anyio
        async def test_postgresql_non_select_query_execution(self, bclearer_postgresql_tool, mock_postgresql_facade):
            """Test executing a non-SELECT query."""
            # Create query input
            query_input = PostgreSQLQueryInput(
                query="UPDATE users SET email = %s WHERE id = %s",
                params=["new_email@example.com", 1]
            )
            
            # Execute the query
            result = await bclearer_postgresql_tool._run(query_input)
            
            # Verify the result
            assert result.success == True
            assert result.row_count == 3  # Number of affected rows from mock
            assert result.results is None
            assert result.column_names is None
            
            # Verify the correct method was called on the facade
            mock_postgresql_facade.execute_non_query.assert_called_once()
        
        @pytest.mark.anyio
        async def test_postgresql_query_execution_with_error(self, bclearer_postgresql_tool, mock_postgresql_facade):
            """Test handling of errors during query execution."""
            # Make the facade raise an exception
            mock_postgresql_facade.execute_query_to_dataframe.side_effect = Exception("Database error")
            
            # Create query input
            query_input = PostgreSQLQueryInput(
                query="SELECT id, username FROM users",
                params=None
            )
            
            # Execute the query
            result = await bclearer_postgresql_tool._run(query_input)
            
            # Verify the result shows failure
            assert result.success == False
            assert "Database error" in result.error_message
            assert result.results is None
            assert result.row_count is None
            assert result.column_names is None

    class TestPostgreSQLSchemaInspectionToolBclearer:
        """Tests for the PostgreSQL schema inspection tool using bclearer implementation."""
        
        @pytest.mark.anyio
        async def test_list_tables_in_schema(self, bclearer_schema_tool, monkeypatch):
            """Test listing tables in a schema."""
            # Create a mock for PostgreSQLToolBclearer._run
            async def mock_run(self, input_data):
                # Import inside the function to avoid circular imports
                from ol_ai_services.tools.database.postgresql_tool_bclearer import PostgreSQLQueryOutput
                return PostgreSQLQueryOutput(
                    success=True,
                    results=[
                        {"table_name": "users"},
                        {"table_name": "products"},
                        {"table_name": "orders"}
                    ],
                    row_count=3,
                    column_names=["table_name"]
                )
            
            # Apply the mock
            monkeypatch.setattr(bclearer_schema_tool.postgresql_tool, "_run", mock_run)
            
            # Create schema inspection input
            schema_input = PostgreSQLSchemaInspectionInput(schema_name="public")
            
            # Execute the schema inspection
            result = await bclearer_schema_tool._run(schema_input)
            
            # Verify the result
            assert result.success == True
            assert result.tables == ["users", "products", "orders"]
            assert result.columns is None
        
        @pytest.mark.anyio
        async def test_describe_table(self, bclearer_schema_tool, monkeypatch):
            """Test describing a table (listing its columns)."""
            # Create a mock for PostgreSQLToolBclearer._run
            async def mock_run(self, input_data):
                # Import inside the function to avoid circular imports
                from ol_ai_services.tools.database.postgresql_tool_bclearer import PostgreSQLQueryOutput
                return PostgreSQLQueryOutput(
                    success=True,
                    results=[
                        {
                            "column_name": "id",
                            "data_type": "integer",
                            "character_maximum_length": None,
                            "column_default": "nextval('users_id_seq'::regclass)",
                            "is_nullable": "NO"
                        },
                        {
                            "column_name": "username",
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
                        }
                    ],
                    row_count=3,
                    column_names=["column_name", "data_type", "character_maximum_length", "column_default", "is_nullable"]
                )
            
            # Apply the mock
            monkeypatch.setattr(bclearer_schema_tool.postgresql_tool, "_run", mock_run)
            
            # Create schema inspection input
            schema_input = PostgreSQLSchemaInspectionInput(
                schema_name="public",
                table_name="users"
            )
            
            # Execute the schema inspection
            result = await bclearer_schema_tool._run(schema_input)
            
            # Verify the result
            assert result.success == True
            assert result.tables is None
            assert len(result.columns) == 3
            
            # Check column details
            assert result.columns[0]["column_name"] == "id"
            assert result.columns[0]["data_type"] == "integer"
            
            assert result.columns[1]["column_name"] == "username"
            assert result.columns[1]["data_type"] == "character varying"
            assert result.columns[1]["character_maximum_length"] == 255