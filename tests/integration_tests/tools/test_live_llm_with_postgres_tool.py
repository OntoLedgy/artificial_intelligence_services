import os
import pytest
import json
import asyncio
from typing import List, Dict, Any
import logging

from ol_ai_services.llms.clients.ollama_clients import OllamaClient
from ol_ai_services.tools.database.postgresql_tool_bclearer import (
    PostgreSQLQueryInput,
    PostgreSQLSchemaInspectionInput,
    PostgreSQLToolBclearer,
    PostgreSQLSchemaInspectionToolBclearer
)
from ol_ai_services.tools.tool_registry import ToolRegistry
from ol_ai_services.tools.tool_integration import ToolIntegration
from tests.fixtures.postgresql import create_test_schema

# Mark all tests in this module as integration tests
pytestmark = [pytest.mark.anyio, pytest.mark.integration]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.skipif(
    not os.environ.get("OLLAMA_API_HOST"), 
    reason="Ollama API host not available in environment"
)
class TestLiveLLMWithPostgreSQLTool:
    """
    Integration tests for using PostgreSQL tools with a real Ollama LLM.
    These tests require both a real Ollama instance and a real PostgreSQL database.
    """
    
    @pytest.mark.anyio
    async def test_llm_using_postgres_query_tool(self, postgresql_tool):
        """
        Test a real Ollama LLM using the PostgreSQL query tool.
        """
        # Create a test schema with tables and data
        async with create_test_schema(postgresql_tool) as schema_name:
            # Register the tools
            registry = ToolRegistry()
            registry.register_tool(postgresql_tool)
            
            # Get the tools in Ollama format
            tools = ToolIntegration.get_tools_for_ollama_format()
            
            # Create the Ollama client
            ollama_host = os.environ.get("OLLAMA_API_HOST", "http://localhost:11434")
            ollama_client = OllamaClient(ollama_api_host=ollama_host)
            
            # Generate a prompt that will make the LLM use the PostgreSQL tool
            prompt = f"""
            A PostgreSQL database has a schema named '{schema_name}' with a 'users' table. 
            The table has columns for id, username, email, and created_at.
            Write a SQL query to select all users from this table and execute it.
            """
            
            # Call the Ollama LLM with the tool
            response = await ollama_client.client.chat(
                model="llama3", 
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                stream=False
            )
            
            # Extract the tool call from the response
            logger.info(f"Ollama response: {response}")
            tool_calls = response.get("message", {}).get("tool_calls", [])
            
            # Verify a tool call was made
            assert len(tool_calls) > 0, "No tool calls were made"
            
            # Find the PostgreSQL query tool call
            pg_tool_call = None
            for tool_call in tool_calls:
                if tool_call.get("name") == "postgresql_query":
                    pg_tool_call = tool_call
                    break
            
            assert pg_tool_call is not None, "No PostgreSQL query tool call found"
            
            # Extract the query from the arguments
            args = json.loads(pg_tool_call.get("arguments", "{}"))
            query = args.get("query", "")
            
            # Verify it's trying to query the users table
            assert f"{schema_name}.users" in query.lower(), f"Query doesn't reference {schema_name}.users table"
            
            # Execute the query using our PostgreSQL tool
            query_input = PostgreSQLQueryInput(query=query)
            result = await postgresql_tool.run(**query_input.dict())
            
            # Verify the query was successful
            assert result.success is True, f"Query failed: {result.error_message}"
            assert result.row_count > 0, "Query returned no results"
            
            # Verify the results contain the expected user data
            assert any(row["username"] == "user1" for row in result.results), "user1 not found in results"
            assert any(row["email"] == "user1@example.com" for row in result.results), "user1's email not found in results"
    
    @pytest.mark.anyio
    async def test_llm_using_schema_inspection(self, postgresql_tool, postgresql_schema_tool):
        """
        Test a real Ollama LLM using the schema inspection tool.
        """
        # Create a test schema with tables and data
        async with create_test_schema(postgresql_tool) as schema_name:
            # Register the tools
            registry = ToolRegistry()
            registry.register_tools([postgresql_tool, postgresql_schema_tool])
            
            # Get the tools in Ollama format
            tools = ToolIntegration.get_tools_for_ollama_format()
            
            # Create the Ollama client
            ollama_host = os.environ.get("OLLAMA_API_HOST", "http://localhost:11434")
            ollama_client = OllamaClient(ollama_api_host=ollama_host)
            
            # Generate a prompt that will make the LLM use the schema inspection tool
            prompt = f"""
            A PostgreSQL database has a schema named '{schema_name}'.
            What tables are in this schema? Use the schema inspection tool to find out.
            """
            
            # Call the Ollama LLM with the tool
            response = await ollama_client.client.chat(
                model="llama3", 
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                stream=False
            )
            
            # Extract the tool call from the response
            logger.info(f"Ollama response: {response}")
            tool_calls = response.get("message", {}).get("tool_calls", [])
            
            # Verify a tool call was made
            assert len(tool_calls) > 0, "No tool calls were made"
            
            # Find the PostgreSQL schema inspection tool call
            schema_tool_call = None
            for tool_call in tool_calls:
                if tool_call.get("name") == "postgresql_schema_inspection":
                    schema_tool_call = tool_call
                    break
            
            assert schema_tool_call is not None, "No PostgreSQL schema inspection tool call found"
            
            # Extract the parameters from the arguments
            args = json.loads(schema_tool_call.get("arguments", "{}"))
            schema_input = PostgreSQLSchemaInspectionInput(**args)
            
            # Verify it's trying to inspect the correct schema
            assert schema_input.schema_name == schema_name, f"Incorrect schema name: {schema_input.schema_name}"
            
            # Execute the schema inspection using our tool
            result = await postgresql_schema_tool.run(**schema_input.dict())
            
            # Verify the inspection was successful
            assert result.success is True, f"Schema inspection failed: {result.error_message}"
            assert result.tables is not None, "No tables returned"
            
            # Verify it found the tables we created
            assert "users" in result.tables, "users table not found"
            assert "products" in result.tables, "products table not found"