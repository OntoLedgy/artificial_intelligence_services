import pytest
import os
from contextlib import asynccontextmanager
import logging

from configurations.ol_configurations.configurations import Configurations
from ol_ai_services.tools.database.postgresql_tool_bclearer import (
    PostgreSQLConnectionConfig,
    PostgreSQLQueryInput,
    PostgreSQLToolBclearer,
    PostgreSQLSchemaInspectionToolBclearer
)


@pytest.fixture(scope="session")
def postgresql_configuration_file_absolute_path(configurations_folder_absolute_path):
    """Returns the absolute path to the PostgreSQL configuration file."""
    postgresql_configuration_file = r"postgresql_configuration.json"
    
    postgresql_configuration_file_absolute_path = os.path.join(
        configurations_folder_absolute_path,
        postgresql_configuration_file
    )
    
    return postgresql_configuration_file_absolute_path


@pytest.fixture(scope="session")
def postgresql_configuration(postgresql_configuration_file_absolute_path):
    """Returns the PostgreSQL configuration object."""
    try:
        configuration_manager = Configurations(postgresql_configuration_file_absolute_path)
        return configuration_manager
    except Exception as e:
        pytest.skip(f"Failed to load PostgreSQL configuration: {str(e)}")


@pytest.fixture(scope="session")
def postgresql_connection_config(postgresql_configuration):
    """Returns a PostgreSQL connection configuration based on the JSON config file."""
    try:
        # Access settings directly instead of using configuration_data
        config = postgresql_configuration.settings.get("POSTGRESQL_CONFIGURATION", {})
        return PostgreSQLConnectionConfig(
            host=config.get("host", "localhost"),
            port=config.get("port", 5432),
            database=config.get("database", "postgres"),
            user=config.get("user", "postgres"),
            password=config.get("password", "postgres")
        )
    except Exception as e:
        pytest.skip(f"Failed to create PostgreSQL connection config: {str(e)}")


@pytest.fixture
def postgresql_tool(postgresql_connection_config):
    """
    Returns a PostgreSQL tool using bclearer_interop_services.
    
    Note: Using function scope instead of session to ensure fresh connections for each test.
    """
    return PostgreSQLToolBclearer(postgresql_connection_config)


@pytest.fixture
def postgresql_schema_tool(postgresql_connection_config):
    """
    Returns a PostgreSQL schema inspection tool using bclearer_interop_services.
    
    Note: Using function scope instead of session to ensure fresh connections for each test.
    """
    return PostgreSQLSchemaInspectionToolBclearer(postgresql_connection_config)


@asynccontextmanager
async def create_test_schema(postgresql_tool, persistent=False):
    """
    Creates a test schema and tables for testing, then cleans it up afterwards.
    
    Args:
        postgresql_tool: The PostgreSQL tool to use for database operations
        persistent: If True, the schema will not be dropped after the test (useful for debugging)
    
    Usage:
    ```
    async with create_test_schema(postgresql_tool) as schema_name:
        # Use the schema_name for testing
    ```
    """
    # Configure logging for this function
    logger = logging.getLogger("test_schema")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    
    logger.debug("Starting test schema creation")
    schema_name = "test_schema"
    
    try:
        # Create schema
        logger.debug(f"Creating schema: {schema_name}")
        create_schema_result = await postgresql_tool.run(
            query=f"CREATE SCHEMA IF NOT EXISTS {schema_name}"
        )
        logger.debug(f"Create schema result: {create_schema_result}")
        
        # Create test users table
        logger.debug(f"Creating users table in schema {schema_name}")
        users_table_result = await postgresql_tool.run(
            query=f"""
            CREATE TABLE IF NOT EXISTS {schema_name}.users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        logger.debug(f"Create users table result: {users_table_result}")
        
        # Check if products table exists and has correct schema
        logger.debug(f"Checking if products table exists with correct schema")
        check_products = await postgresql_tool.run(
            query=f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{schema_name}' AND table_name = 'products'
            """
        )
        logger.debug(f"Products table columns: {check_products}")
        
        # Drop the products table if it exists to create it with the correct schema
        logger.debug(f"Dropping products table if it exists")
        drop_products = await postgresql_tool.run(
            query=f"DROP TABLE IF EXISTS {schema_name}.products CASCADE"
        )
        logger.debug(f"Drop products table result: {drop_products}")
        
        # Create test products table
        logger.debug(f"Creating products table in schema {schema_name}")
        products_table_result = await postgresql_tool.run(
            query=f"""
            CREATE TABLE IF NOT EXISTS {schema_name}.products (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                price NUMERIC(10, 2) NOT NULL,
                description TEXT
            )
            """
        )
        logger.debug(f"Create products table result: {products_table_result}")
        
        # Insert test data into users table
        logger.debug(f"Inserting test data into users table")
        users_data_result = await postgresql_tool.run(
            query=f"""
            INSERT INTO {schema_name}.users (username, email)
            VALUES 
                ('user1', 'user1@example.com'),
                ('user2', 'user2@example.com'),
                ('user3', 'user3@example.com')
            ON CONFLICT (id) DO NOTHING
            """
        )
        logger.debug(f"Insert users data result: {users_data_result}")
        
        # Insert test data into products table
        logger.debug(f"Inserting test data into products table")
        products_data_result = await postgresql_tool.run(
            query=f"""
            INSERT INTO {schema_name}.products (name, price, description)
            VALUES 
                ('Product 1', 19.99, 'Description for product 1'),
                ('Product 2', 29.99, 'Description for product 2'),
                ('Product 3', 39.99, 'Description for product 3')
            ON CONFLICT (id) DO NOTHING
            """
        )
        logger.debug(f"Insert products data result: {products_data_result}")
        
        # Force a commit to ensure data is visible to other connections
        logger.debug("Committing transaction")
        commit_result = await postgresql_tool.run(query="COMMIT")
        logger.debug(f"Commit result: {commit_result}")
        
        # Verify schema exists before yielding
        logger.debug(f"Verifying schema {schema_name} exists")
        verify_result = await postgresql_tool.run(
            query=f"""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name = '{schema_name}'
            """
        )
        logger.debug(f"Schema verification result: {verify_result}")
        
        # Check tables in the schema
        tables_result = await postgresql_tool.run(
            query=f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = '{schema_name}'
            """
        )
        logger.debug(f"Tables in schema {schema_name}: {tables_result}")
        
        # Check data in users table
        users_data_check = await postgresql_tool.run(
            query=f"SELECT COUNT(*) FROM {schema_name}.users"
        )
        logger.debug(f"Users table row count: {users_data_check}")
        
        # If users table is empty, try rerunning the insert
        if users_data_check.results and users_data_check.results[0]['count'] == 0:
            logger.debug("Users table is empty, re-inserting data")
            reinsert_users = await postgresql_tool.run(
                query=f"""
                INSERT INTO {schema_name}.users (username, email)
                VALUES 
                    ('user1', 'user1@example.com'),
                    ('user2', 'user2@example.com'),
                    ('user3', 'user3@example.com')
                ON CONFLICT (id) DO NOTHING
                """
            )
            logger.debug(f"Re-insert users data result: {reinsert_users}")
            
            # Verify the insert worked
            verify_users = await postgresql_tool.run(
                query=f"SELECT * FROM {schema_name}.users"
            )
            logger.debug(f"Users table after re-insert: {verify_users}")
        
        # Return the schema name to use in tests
        logger.debug(f"Returning schema name: {schema_name}")
        yield schema_name
        
    finally:
        # Clean up - drop the schema and all objects in it unless persistent is True
        if not persistent:
            try:
                logger.debug(f"Dropping schema {schema_name}")
                drop_result = await postgresql_tool.run(
                    query=f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"
                )
                logger.debug(f"Drop schema result: {drop_result}")
            except Exception as e:
                logger.error(f"Error dropping schema {schema_name}: {str(e)}")
        else:
            logger.debug(f"Schema {schema_name} kept due to persistent=True flag")
            print(f"NOTE: Test schema '{schema_name}' was not dropped due to persistent=True flag")