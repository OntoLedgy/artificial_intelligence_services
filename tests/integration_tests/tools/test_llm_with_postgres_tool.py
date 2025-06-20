import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from ol_ai_services.llms.clients.open_ai_clients import OpenAiClients
from ol_ai_services.tools.database.postgresql_tool import (
    PostgreSQLQueryInput,
    PostgreSQLSchemaInspectionInput
)
from ol_ai_services.tools.tool_registry import ToolRegistry
from ol_ai_services.tools.tool_integration import ToolIntegration

# Mark all tests in this module as integration tests
pytestmark = [pytest.mark.anyio, pytest.mark.integration]

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), 
    reason="OpenAI API key not available in environment"
)
class TestLLMWithPostgreSQLTool:
    """Integration tests for using PostgreSQL tools with an LLM."""
    
    async def test_llm_using_postgres_query_tool(self, mock_sqlite_adapter, integration_postgresql_tool):
        """
        Test an LLM using the PostgreSQL query tool with our SQLite adapter.
        """
        # Create and register the tool
        pg_tool = integration_postgresql_tool(mock_sqlite_adapter)
        registry = ToolRegistry()
        registry._tools = {}  # Clear the registry
        registry.register_tool(pg_tool)
        
        # Mock the OpenAI client response
        mock_client = MagicMock()
        mock_client.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message={
                        "role": "assistant",
                        "content": "I'll query the users table for you.",
                        "function_call": {
                            "name": "postgresql_query",
                            "arguments": '{"query": "SELECT * FROM users"}'
                        }
                    },
                    finish_reason="function_call"
                )
            ]
        )
        
        # Patch the OpenAI client
        with patch("openai.ChatCompletion", mock_client):
            # Create the OpenAI client
            api_key = os.environ.get("OPENAI_API_KEY", "fake_key_for_testing")
            open_ai_client = OpenAiClients(api_key=api_key)
            
            # Get the tools in OpenAI format
            tools = ToolIntegration.get_tools_for_openai_format()
            
            # Simulate the LLM call with function calling
            response = open_ai_client.client.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Show me all users in the database"}],
                tools=tools,
                temperature=0
            )
            
            # Extract the function call
            function_call = response.choices[0].message["function_call"]
            assert function_call["name"] == "postgresql_query"
            
            # Execute the query as if the LLM called it
            import json
            query_input = PostgreSQLQueryInput(**json.loads(function_call["arguments"]))
            result = await pg_tool.run(**query_input.dict())
            
            # Verify the result
            assert result.success is True
            assert len(result.results) == 3
            assert result.results[0]["id"] == 1
            assert result.results[0]["username"] == "user1"
            assert result.results[0]["email"] == "user1@example.com"
    
    async def test_llm_using_schema_inspection_tool(self, mock_sqlite_adapter, integration_postgresql_tool, integration_schema_tool):
        """
        Test an LLM using the schema inspection tool with our SQLite adapter.
        """
        # Create and register the tools
        pg_tool = integration_postgresql_tool(mock_sqlite_adapter)
        schema_tool = integration_schema_tool(pg_tool)
        
        registry = ToolRegistry()
        registry._tools = {}  # Clear the registry
        registry.register_tools([pg_tool, schema_tool])
        
        # Mock the OpenAI client response
        mock_client = MagicMock()
        mock_client.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message={
                        "role": "assistant",
                        "content": "I'll check what tables are in the database.",
                        "function_call": {
                            "name": "postgresql_schema_inspection",
                            "arguments": '{"schema_name": "public"}'
                        }
                    },
                    finish_reason="function_call"
                )
            ]
        )
        
        # Patch the OpenAI client
        with patch("openai.ChatCompletion", mock_client):
            # Create the OpenAI client
            api_key = os.environ.get("OPENAI_API_KEY", "fake_key_for_testing")
            open_ai_client = OpenAiClients(api_key=api_key)
            
            # Get the tools in OpenAI format
            tools = ToolIntegration.get_tools_for_openai_format()
            
            # Simulate the LLM call with function calling
            response = open_ai_client.client.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "What tables are in the database?"}],
                tools=tools,
                temperature=0
            )
            
            # Extract the function call
            function_call = response.choices[0].message["function_call"]
            assert function_call["name"] == "postgresql_schema_inspection"
            
            # Execute the schema inspection as if the LLM called it
            import json
            schema_input = PostgreSQLSchemaInspectionInput(**json.loads(function_call["arguments"]))
            result = await schema_tool.run(**schema_input.dict())
            
            # Verify the result
            assert result.success is True
            assert "users" in result.tables
            assert "products" in result.tables
    
    async def test_schema_inspection_followed_by_query(self, mock_sqlite_adapter, integration_postgresql_tool, integration_schema_tool):
        """
        Test an LLM using the schema inspection tool followed by a query tool.
        This simulates a more realistic interaction where the LLM first explores the schema
        and then uses that information to formulate a query.
        """
        # Create and register the tools
        pg_tool = integration_postgresql_tool(mock_sqlite_adapter)
        schema_tool = integration_schema_tool(pg_tool)
        
        registry = ToolRegistry()
        registry._tools = {}  # Clear the registry
        registry.register_tools([pg_tool, schema_tool])
        
        # Mock the OpenAI client responses for schema inspection
        mock_client = MagicMock()
        
        # First response - schema inspection
        mock_client.create.side_effect = [
            # First call - return schema inspection
            MagicMock(
                choices=[
                    MagicMock(
                        message={
                            "role": "assistant",
                            "content": "Let me check the database schema first.",
                            "function_call": {
                                "name": "postgresql_schema_inspection",
                                "arguments": '{"schema_name": "public"}'
                            }
                        },
                        finish_reason="function_call"
                    )
                ]
            ),
            # Second call - return table inspection
            MagicMock(
                choices=[
                    MagicMock(
                        message={
                            "role": "assistant",
                            "content": "Let me check the products table structure.",
                            "function_call": {
                                "name": "postgresql_schema_inspection",
                                "arguments": '{"schema_name": "public", "table_name": "products"}'
                            }
                        },
                        finish_reason="function_call"
                    )
                ]
            ),
            # Third call - return query
            MagicMock(
                choices=[
                    MagicMock(
                        message={
                            "role": "assistant",
                            "content": "Now I'll query products with price > 20.",
                            "function_call": {
                                "name": "postgresql_query",
                                "arguments": '{"query": "SELECT * FROM products WHERE price > 20"}'
                            }
                        },
                        finish_reason="function_call"
                    )
                ]
            )
        ]
        
        # Patch the OpenAI client
        with patch("openai.ChatCompletion", mock_client):
            # Create the OpenAI client
            api_key = os.environ.get("OPENAI_API_KEY", "fake_key_for_testing")
            open_ai_client = OpenAiClients(api_key=api_key)
            
            # Get the tools in OpenAI format
            tools = ToolIntegration.get_tools_for_openai_format()
            
            # Simulate the conversation with the LLM
            
            # Step 1: List tables
            response1 = open_ai_client.client.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "What tables are in the database?"}],
                tools=tools,
                temperature=0
            )
            
            function_call1 = response1.choices[0].message["function_call"]
            assert function_call1["name"] == "postgresql_schema_inspection"
            
            schema_input1 = PostgreSQLSchemaInspectionInput(**json.loads(function_call1["arguments"]))
            result1 = await schema_tool.run(**schema_input1.dict())
            
            # Step 2: Inspect products table
            response2 = open_ai_client.client.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": "What tables are in the database?"},
                    {"role": "assistant", "content": "Let me check the database schema first."},
                    {"role": "function", "name": "postgresql_schema_inspection", 
                     "content": f"Tables: {', '.join(result1.tables)}"},
                    {"role": "user", "content": "What's in the products table?"}
                ],
                tools=tools,
                temperature=0
            )
            
            function_call2 = response2.choices[0].message["function_call"]
            assert function_call2["name"] == "postgresql_schema_inspection"
            
            schema_input2 = PostgreSQLSchemaInspectionInput(**json.loads(function_call2["arguments"]))
            result2 = await schema_tool.run(**schema_input2.dict())
            
            # Step 3: Query products with price > 20
            response3 = open_ai_client.client.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": "What tables are in the database?"},
                    {"role": "assistant", "content": "Let me check the database schema first."},
                    {"role": "function", "name": "postgresql_schema_inspection", 
                     "content": f"Tables: {', '.join(result1.tables)}"},
                    {"role": "user", "content": "What's in the products table?"},
                    {"role": "assistant", "content": "Let me check the products table structure."},
                    {"role": "function", "name": "postgresql_schema_inspection", 
                     "content": f"Columns: {', '.join([col['column_name'] for col in result2.columns])}"},
                    {"role": "user", "content": "Show me products with price greater than 20"}
                ],
                tools=tools,
                temperature=0
            )
            
            function_call3 = response3.choices[0].message["function_call"]
            assert function_call3["name"] == "postgresql_query"
            
            query_input = PostgreSQLQueryInput(**json.loads(function_call3["arguments"]))
            result3 = await pg_tool.run(**query_input.dict())
            
            # Verify the final query result
            assert result3.success is True
            # Should return products 2 and 3 which have price > 20
            assert len(result3.results) == 2
            assert result3.results[0]["price"] > 20
            assert result3.results[1]["price"] > 20
            assert result3.results[0]["name"] == "Product 2"
            assert result3.results[1]["name"] == "Product 3"