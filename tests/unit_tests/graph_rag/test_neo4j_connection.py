import pytest
from neo4j import GraphDatabase

def test_neo4j_basic_connection():
    # Connection parameters
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "Numark234"
    
    try:
        # Create a driver instance
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Verify connectivity
        with driver.session() as session:
            result = session.run("RETURN 1 as num")
            record = result.single()
            assert record["num"] == 1, "Failed to get expected result from Neo4j"
            
            print("Neo4j connection test successful")
            
            # Get Neo4j version
            version_result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions")
            for record in version_result:
                if record["name"] == "Neo4j Kernel":
                    print(f"Neo4j version: {record['versions']}")
            
            # Test APOC availability
            try:
                apoc_result = session.run("CALL apoc.help('apoc')")
                print(f"APOC procedures available: {len(list(apoc_result))} procedures")
            except Exception as e:
                print(f"APOC test failed: {e}")
            
            # Test a simple CALL subquery syntax similar to what llama-index might be using
            try:
                call_result = session.run("""
                MATCH (n) 
                WITH n LIMIT 1
                CALL {
                    WITH n
                    RETURN n.name as test_name
                }
                RETURN test_name
                """)
                print(f"CALL subquery test result: {list(call_result)}")
            except Exception as e:
                print(f"CALL subquery test failed: {e}")
                
            # Test the problematic CALL syntax that's causing the error
            try:
                problematic_call = session.run("""
                MATCH (n)
                WITH n LIMIT 1
                CALL (e, row) {
                    WITH n
                    RETURN n.name as test_name
                }
                RETURN test_name
                """)
                print(f"Problematic CALL test result: {list(problematic_call)}")
            except Exception as e:
                print(f"Problematic CALL syntax test failed with expected error: {e}")
                
            # Test APOC procedure call
            try:
                apoc_call = session.run("CALL apoc.create.node(['Test'], {name: 'TestNode'})")
                print(f"APOC node creation test result: {list(apoc_call)}")
            except Exception as e:
                print(f"APOC node creation test failed: {e}")
                
    except Exception as e:
        pytest.fail(f"Neo4j connection test failed: {e}")
    finally:
        # Close the driver
        if 'driver' in locals():
            driver.close()
