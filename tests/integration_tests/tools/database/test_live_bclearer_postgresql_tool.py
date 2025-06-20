import pytest
from typing import Dict, Any, List, Optional

from ol_ai_services.tools.database.postgresql_tool_bclearer import (
    PostgreSQLConnectionConfig,
    PostgreSQLQueryInput,
    PostgreSQLSchemaInspectionInput,
    PostgreSQLToolBclearer,
    PostgreSQLSchemaInspectionToolBclearer
)
from tests.fixtures.postgresql import create_test_schema


class TestBClearerPostgreSQLToolRealDatabase:
    """Tests for bclearer PostgreSQL tools with a real database connection."""
    
    @pytest.mark.anyio
    async def test_connection_to_real_database(self, postgresql_tool):
        """Test that the tool can connect to the real database."""
        # Simply test a simple query like getting the database version
        query_input = PostgreSQLQueryInput(query="SELECT version()")
        result = await postgresql_tool.run(**query_input.dict())
        
        assert result.success is True
        assert len(result.results) > 0
        assert "version" in result.results[0]
    
    @pytest.mark.anyio
    async def test_create_and_query_tables(self, postgresql_tool):
        """Test creating tables, inserting data, and querying it."""
        async with create_test_schema(postgresql_tool) as schema_name:
            # Query the users table
            query_input = PostgreSQLQueryInput(
                query=f"SELECT * FROM {schema_name}.users ORDER BY id"
            )
            result = await postgresql_tool.run(**query_input.dict())
            
            # Verify the result
            assert result.success is True
            assert result.row_count == 3
            assert len(result.results) == 3
            assert result.results[0]["username"] == "user1"
            assert result.results[0]["email"] == "user1@example.com"
            
            # Query with parameters
            param_query_input = PostgreSQLQueryInput(
                query=f"SELECT * FROM {schema_name}.users WHERE username = %s",
                params=["user2"]
            )
            param_result = await postgresql_tool.run(**param_query_input.dict())
            
            # Verify the parameter query result
            assert param_result.success is True
            assert param_result.row_count == 1
            assert param_result.results[0]["email"] == "user2@example.com"
    
    @pytest.mark.anyio
    async def test_update_data(self, postgresql_tool):
        """Test updating data in the database."""
        async with create_test_schema(postgresql_tool) as schema_name:
            # Update the email for user1
            update_input = PostgreSQLQueryInput(
                query=f"UPDATE {schema_name}.users SET email = %s WHERE username = %s",
                params=["updated@example.com", "user1"]
            )
            update_result = await postgresql_tool.run(**update_input.dict())
            
            # Verify the update was successful
            assert update_result.success is True
            assert update_result.row_count == 1  # One row updated
            
            # Query to confirm the update
            query_input = PostgreSQLQueryInput(
                query=f"SELECT email FROM {schema_name}.users WHERE username = %s",
                params=["user1"]
            )
            query_result = await postgresql_tool.run(**query_input.dict())
            
            # Verify the query result shows the updated email
            assert query_result.success is True
            assert query_result.results[0]["email"] == "updated@example.com"
    
    @pytest.mark.anyio
    async def test_schema_inspection(self, postgresql_schema_tool, postgresql_tool):
        """Test inspecting the database schema."""
        async with create_test_schema(postgresql_tool) as schema_name:
            # List tables in the schema
            schema_input = PostgreSQLSchemaInspectionInput(
                schema_name=schema_name
            )
            result = await postgresql_schema_tool.run(**schema_input.dict())
            
            # Verify the tables are listed
            assert result.success is True
            assert len(result.tables) >= 2
            assert "users" in result.tables
            assert "products" in result.tables
            
            # Describe the users table
            table_input = PostgreSQLSchemaInspectionInput(
                schema_name=schema_name,
                table_name="users"
            )
            table_result = await postgresql_schema_tool.run(**table_input.dict())
            
            # Verify the column details
            assert table_result.success is True
            assert len(table_result.columns) == 4
            
            # Find the username column
            username_col = next(col for col in table_result.columns if col["column_name"] == "username")
            assert username_col["data_type"] == "character varying"
            assert username_col["character_maximum_length"] == 255