import pytest
import sqlite3
import os
import tempfile
from typing import Dict, Any
from contextlib import closing

from ol_ai_services.tools.database.postgresql_tool_bclearer import (
    PostgreSQLConnectionConfig,
    PostgreSQLToolBclearer,
    PostgreSQLSchemaInspectionToolBclearer,
    PostgreSQLSchemaInspectionOutput
)

class SQLitePostgreSQLAdapter:
    """
    Adapter class that makes SQLite behave like PostgreSQL for testing purposes.
    This allows integration tests to run without a real PostgreSQL database.
    """
    
    def __init__(self):
        # Create a temporary SQLite database
        self.db_file = tempfile.NamedTemporaryFile(delete=False).name
        # Use check_same_thread=False to allow SQLite to be used in multiple threads
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.setup_test_database()
        
    def setup_test_database(self):
        """Set up test tables and data in the SQLite database."""
        with closing(self.conn.cursor()) as cursor:
            # Create users table
            cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                email TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Insert sample data
            cursor.executemany(
                "INSERT INTO users (id, username, email) VALUES (?, ?, ?)",
                [
                    (1, "user1", "user1@example.com"),
                    (2, "user2", "user2@example.com"),
                    (3, "user3", "user3@example.com"),
                ]
            )
            
            # Create products table
            cursor.execute("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                description TEXT
            )
            """)
            
            # Insert sample data
            cursor.executemany(
                "INSERT INTO products (id, name, price, description) VALUES (?, ?, ?, ?)",
                [
                    (1, "Product 1", 19.99, "Description for product 1"),
                    (2, "Product 2", 29.99, "Description for product 2"),
                    (3, "Product 3", 39.99, "Description for product 3"),
                ]
            )
            
            self.conn.commit()
    
    def fetch_results(self, query, params=None):
        """Execute a query and return a pandas DataFrame of results."""
        import pandas as pd
        import logging
        
        try:
            # Convert params if needed
            if params and not isinstance(params, (list, tuple)):
                params = [params]
                
            # Execute the query and return results as a DataFrame
            return pd.read_sql_query(query, self.conn, params=params)
        except Exception as e:
            logging.error(f"Error in fetch_results: {str(e)}")
            raise
    
    def execute_query(self, query, params=None):
        """Execute a non-SELECT query."""
        import logging
        
        try:
            # Convert params if needed
            if params and not isinstance(params, (list, tuple)):
                params = [params]
                
            # Execute the query
            cursor = self.conn.cursor()
            cursor.execute(query, params or [])
            self.conn.commit()
            cursor.close()
            return True
        except Exception as e:
            logging.error(f"Error in execute_query: {str(e)}")
            raise
        
    def connect(self):
        """Connect to the database - no-op for SQLite as we're already connected."""
        pass
        
    def disconnect(self):
        """Disconnect from the database - no-op for SQLite in testing context."""
        pass
    
    def close(self):
        """Close the database connection and remove the temporary file."""
        try:
            if self.conn:
                self.conn.close()
            if os.path.exists(self.db_file):
                os.unlink(self.db_file)
        except Exception as e:
            import logging
            logging.error(f"Error closing SQLite adapter: {str(e)}")
    
    def get_connection_config(self) -> Dict[str, Any]:
        """Return a connection configuration that points to this SQLite database."""
        return {
            "host": "localhost",
            "port": 5432,
            "database": self.db_file,
            "user": "test_user",
            "password": "test_password"
        }


@pytest.fixture
def mock_sqlite_adapter():
    """Return an SQLitePostgreSQLAdapter instance."""
    adapter = SQLitePostgreSQLAdapter()
    yield adapter
    adapter.close()


@pytest.fixture
def integration_postgresql_tool(monkeypatch):
    """Return a PostgreSQL tool that works with SQLite for integration testing."""
    # This will be set by tests
    sqlite_adapter = None
    
    # Return a mock facade object that mimics the PostgresqlFacade
    class MockFacade:
        def __init__(self, adapter):
            self.adapter = adapter
            self.conn = adapter.conn
            
        def fetch_results(self, query, params=None):
            return self.adapter.fetch_results(query, params)
            
        def execute_query(self, query, params=None):
            return self.adapter.execute_query(query, params)
            
        def connect(self):
            self.adapter.connect()
            
        def disconnect(self):
            self.adapter.disconnect()
    
    # Mock the connection to use SQLite instead
    async def mock_connect(self):
        self._facade = MockFacade(sqlite_adapter)
    
    async def mock_disconnect(self):
        pass
    
    # Return a factory function that accepts the adapter
    def _create_tool(adapter):
        nonlocal sqlite_adapter
        sqlite_adapter = adapter
        
        config = PostgreSQLConnectionConfig(**adapter.get_connection_config())
        tool = PostgreSQLToolBclearer(config)
        
        # Replace the connection method
        monkeypatch.setattr(tool, "_connect", lambda self=tool: mock_connect(self))
        monkeypatch.setattr(tool, "_disconnect", lambda self=tool: mock_disconnect(self))
        
        return tool
    
    return _create_tool


@pytest.fixture
def integration_schema_tool(monkeypatch):
    """Return a schema inspection tool that works with SQLite for integration testing."""
    # This will be set by tests
    tool = None
    
    # Mock the query execution to work with SQLite's schema tables
    async def mock_run(self, input_data):
        # For table listing
        if input_data.table_name is None:
            try:
                # Use the adapter's connection through the facade
                cursor = self.postgresql_tool._facade.conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                tables = [row[0] for row in cursor.fetchall()]
                
                return PostgreSQLSchemaInspectionOutput(
                    success=True,
                    tables=tables,
                    columns=None
                )
            except Exception as e:
                import logging
                logging.error(f"Error in schema inspection: {str(e)}")
                return PostgreSQLSchemaInspectionOutput(
                    success=False,
                    error_message=f"Schema inspection error: {str(e)}",
                    tables=None,
                    columns=None
                )
        # For column inspection
        else:
            try:
                cursor = self.postgresql_tool._facade.conn.cursor()
                cursor.execute(f"PRAGMA table_info({input_data.table_name})")
                columns = []
                for row in cursor.fetchall():
                    columns.append({
                        "column_name": row[1],
                        "data_type": row[2],
                        "character_maximum_length": None,
                        "column_default": row[4] if row[4] != "NULL" else None,
                        "is_nullable": "NO" if row[3] else "YES"
                    })
                
                return PostgreSQLSchemaInspectionOutput(
                    success=True,
                    tables=None,
                    columns=columns
                )
            except Exception as e:
                import logging
                logging.error(f"Error in column inspection: {str(e)}")
                return PostgreSQLSchemaInspectionOutput(
                    success=False,
                    error_message=f"Column inspection error: {str(e)}",
                    tables=None,
                    columns=None
                )
    
    # Return a factory function that accepts the tool
    def _create_tool(postgresql_tool):
        nonlocal tool
        
        tool = PostgreSQLSchemaInspectionToolBclearer(postgresql_tool.connection_config)
        tool.postgresql_tool = postgresql_tool
        
        # Replace the run method
        monkeypatch.setattr(tool, "_run", lambda self_param=tool, input_data=None: mock_run(self_param, input_data))
        
        return tool
    
    return _create_tool