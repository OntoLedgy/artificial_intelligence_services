import asyncio
from typing import List, Dict, Any, Optional, Union
from pydantic import Field, validator
import logging

from ol_ai_services.tools.base_tool import BaseTool, ToolInput, ToolOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from bclearer_interop_services.relational_database_services.postgresql.PostgresqlFacade import PostgresqlFacade

class PostgreSQLConnectionConfig(ToolInput):
    """Configuration for PostgreSQL connection."""
    host: str = Field(..., description="PostgreSQL server host")
    port: int = Field(default=5432, description="PostgreSQL server port")
    database: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    
    @validator("port")
    def validate_port(cls, v):
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class PostgreSQLQueryInput(ToolInput):
    """Input for PostgreSQL query execution."""
    query: str = Field(..., description="SQL query to execute")
    params: Optional[List[Any]] = Field(default=None, description="Parameters for the query")


class PostgreSQLQueryOutput(ToolOutput):
    """Output for PostgreSQL query execution."""
    results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Query results")
    row_count: Optional[int] = Field(default=None, description="Number of rows affected or returned")
    column_names: Optional[List[str]] = Field(default=None, description="Column names in the result set")


class PostgreSQLToolBclearer(BaseTool):
    """
    Tool for connecting to PostgreSQL and executing queries using bclearer_interop_services.
    
    This tool allows LLMs to connect to a PostgreSQL database and execute
    SQL queries against it.
    """
    
    input_schema = PostgreSQLQueryInput
    output_schema = PostgreSQLQueryOutput
    
    def __init__(self, connection_config: PostgreSQLConnectionConfig):
        """
        Initialize the PostgreSQL tool with connection configuration.
        
        Args:
            connection_config: The configuration for connecting to the PostgreSQL server
        """
        super().__init__(
            name="postgresql_query",
            description="Execute SQL queries against a PostgreSQL database"
        )
        self.connection_config = connection_config
        self._facade = None
        

    
    async def _connect(self):
        """
        Connect to the PostgreSQL server using bclearer_interop_services.
        """
        # Always create a new connection to avoid transaction isolation issues
        # Use asyncio to run the blocking operation in a thread pool
        loop = asyncio.get_event_loop()
        # Create facade and explicitly connect to the database
        facade = PostgresqlFacade(
            host=self.connection_config.host,
            database=self.connection_config.database,
            user=self.connection_config.user,
            password=self.connection_config.password,
        )
        
        # Explicitly connect (PostgresqlFacade doesn't auto-connect in constructor)
        await loop.run_in_executor(None, facade.connect)
        
        self._facade = facade
        
        # Log connection created for debugging
        logger.info(f"PostgreSQL connection created to {self.connection_config.host}:{self.connection_config.database}")
        
    async def _disconnect(self):
        """
        Disconnect from the PostgreSQL server and clean up resources.
        """
        if self._facade is not None:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._facade.disconnect)
                logger.info(f"PostgreSQL connection closed to {self.connection_config.host}:{self.connection_config.database}")
            except Exception as e:
                logger.error(f"Error disconnecting from PostgreSQL: {str(e)}")
            finally:
                self._facade = None
    
    async def _run(self, input_data: PostgreSQLQueryInput) -> PostgreSQLQueryOutput:
        """
        Execute a SQL query against the PostgreSQL database.
        
        Args:
            input_data: The query to execute and its parameters
            
        Returns:
            The query results
        """
        try:
            await self._connect()
            
            query = input_data.query
            params = input_data.params if input_data.params else []
            
            # Determine query type by examining the first word
            query_type = query.strip().split(' ')[0].upper()
            
            # Use asyncio to run the blocking database operations in a thread pool
            loop = asyncio.get_event_loop()
            
            try:
                # For SELECT queries
                if query_type == 'SELECT':
                    # Execute query and get results
                    # Use fetch_results from PostgresqlFacade instead of execute_query_to_dataframe
                    results = await loop.run_in_executor(
                        None, 
                        lambda: self._facade.fetch_results(query, params)
                    )
                    
                    # Convert DataFrame to list of dictionaries
                    if results is not None:
                        records = results.to_dict('records')
                        column_names = list(results.columns)
                        
                        return PostgreSQLQueryOutput(
                            success=True,
                            results=records,
                            row_count=len(records),
                            column_names=column_names
                        )
                    else:
                        return PostgreSQLQueryOutput(
                            success=True,
                            results=[],
                            row_count=0,
                            column_names=[]
                        )
                else:
                    # For non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
                    await loop.run_in_executor(
                        None, 
                        lambda: self._facade.execute_query(query, params)
                    )
                    # PostgresqlFacade doesn't return affected rows, so assume 1
                    affected_rows = 1
                    
                    return PostgreSQLQueryOutput(
                        success=True,
                        results=None,
                        row_count=affected_rows,
                        column_names=None
                    )
            finally:
                # Always ensure connection is closed
                await self._disconnect()
                
        except Exception as e:
            logger.error(f"Error executing PostgreSQL query: {str(e)}")
            # Ensure connection is closed even on error
            await self._disconnect()
            return PostgreSQLQueryOutput(
                success=False,
                error_message=f"Database error: {str(e)}",
                results=None,
                row_count=None,
                column_names=None
            )


class PostgreSQLSchemaInspectionInput(ToolInput):
    """Input for PostgreSQL schema inspection."""
    schema_name: Optional[str] = Field(default="public", description="Schema name to inspect")
    table_name: Optional[str] = Field(default=None, description="Table name to inspect (if None, list all tables)")


class PostgreSQLSchemaInspectionOutput(ToolOutput):
    """Output for PostgreSQL schema inspection."""
    tables: Optional[List[str]] = Field(default=None, description="List of tables in the schema")
    columns: Optional[List[Dict[str, Any]]] = Field(default=None, description="Columns in the specified table")


class PostgreSQLSchemaInspectionToolBclearer(BaseTool):
    """
    Tool for inspecting PostgreSQL database schema using bclearer_interop_services.
    
    This tool allows LLMs to inspect the structure of a PostgreSQL database, listing
    tables and viewing column definitions.
    """
    
    input_schema = PostgreSQLSchemaInspectionInput
    output_schema = PostgreSQLSchemaInspectionOutput
    
    def __init__(self, connection_config: PostgreSQLConnectionConfig):
        """
        Initialize the PostgreSQL schema inspection tool.
        
        Args:
            connection_config: The configuration for connecting to the PostgreSQL server
        """
        super().__init__(
            name="postgresql_schema_inspection",
            description="Inspect the schema of a PostgreSQL database"
        )
        self.connection_config = connection_config
        self.postgresql_tool = PostgreSQLToolBclearer(connection_config)
    
    async def _run(self, input_data: PostgreSQLSchemaInspectionInput) -> PostgreSQLSchemaInspectionOutput:
        """
        Inspect the schema of the PostgreSQL database.
        
        Args:
            input_data: The schema inspection parameters
            
        Returns:
            Information about the database schema
        """
        try:
            if input_data.table_name is None:
                # List all tables in the schema
                query = f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s
                ORDER BY table_name
                """
                
                query_input = PostgreSQLQueryInput(query=query, params=[input_data.schema_name])
                result = await self.postgresql_tool._run(query_input)
                
                if not result.success:
                    return PostgreSQLSchemaInspectionOutput(
                        success=False,
                        error_message=result.error_message
                    )
                
                tables = [row["table_name"] for row in result.results] if result.results else []
                
                return PostgreSQLSchemaInspectionOutput(
                    success=True,
                    tables=tables,
                    columns=None
                )
            else:
                # Get column information for the specified table
                query = f"""
                SELECT 
                    column_name, 
                    data_type, 
                    character_maximum_length, 
                    column_default, 
                    is_nullable 
                FROM information_schema.columns 
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """
                
                query_input = PostgreSQLQueryInput(
                    query=query, 
                    params=[input_data.schema_name, input_data.table_name]
                )
                result = await self.postgresql_tool._run(query_input)
                
                if not result.success:
                    return PostgreSQLSchemaInspectionOutput(
                        success=False,
                        error_message=result.error_message
                    )
                
                return PostgreSQLSchemaInspectionOutput(
                    success=True,
                    tables=None,
                    columns=result.results
                )
        except Exception as e:
            logger.error(f"Error inspecting PostgreSQL schema: {str(e)}")
            return PostgreSQLSchemaInspectionOutput(
                success=False,
                error_message=f"Schema inspection error: {str(e)}",
                tables=None,
                columns=None
            )