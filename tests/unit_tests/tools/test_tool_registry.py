import pytest
from ol_ai_services.tools.base_tool import BaseTool, ToolInput, ToolOutput
from ol_ai_services.tools.tool_registry import ToolRegistry
from pydantic import Field

# Define test tool classes
class TestToolInput(ToolInput):
    parameter: str = Field(..., description="Test parameter")

class TestToolOutput(ToolOutput):
    result: str = Field(..., description="Test result")

class TestTool(BaseTool):
    """Tool for testing purposes."""
    
    input_schema = TestToolInput
    output_schema = TestToolOutput
    
    async def _run(self, input_data: TestToolInput) -> TestToolOutput:
        return TestToolOutput(
            success=True,
            result=f"Processed: {input_data.parameter}"
        )

class TestToolRegistry:
    """Tests for the ToolRegistry class."""
    
    def test_singleton_pattern(self):
        """Test that ToolRegistry is a singleton."""
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()
        
        assert registry1 is registry2
    
    def test_register_tool(self):
        """Test registering a single tool."""
        registry = ToolRegistry()
        registry._tools = {}  # Clear the registry
        
        tool = TestTool(name="test_tool", description="A test tool")
        registry.register_tool(tool)
        
        assert "test_tool" in registry._tools
        assert registry._tools["test_tool"] is tool
    
    def test_register_tools(self):
        """Test registering multiple tools at once."""
        registry = ToolRegistry()
        registry._tools = {}  # Clear the registry
        
        tool1 = TestTool(name="test_tool_1", description="Test tool 1")
        tool2 = TestTool(name="test_tool_2", description="Test tool 2")
        
        registry.register_tools([tool1, tool2])
        
        assert "test_tool_1" in registry._tools
        assert "test_tool_2" in registry._tools
        assert registry._tools["test_tool_1"] is tool1
        assert registry._tools["test_tool_2"] is tool2
    
    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        registry._tools = {}  # Clear the registry
        
        tool = TestTool(name="test_tool", description="A test tool")
        registry.register_tool(tool)
        
        retrieved_tool = registry.get_tool("test_tool")
        assert retrieved_tool is tool
        
        # Test getting a non-existent tool
        non_existent_tool = registry.get_tool("non_existent_tool")
        assert non_existent_tool is None
    
    def test_get_all_tools(self):
        """Test getting all registered tools."""
        registry = ToolRegistry()
        registry._tools = {}  # Clear the registry
        
        tool1 = TestTool(name="test_tool_1", description="Test tool 1")
        tool2 = TestTool(name="test_tool_2", description="Test tool 2")
        
        registry.register_tools([tool1, tool2])
        
        all_tools = registry.get_all_tools()
        assert len(all_tools) == 2
        assert tool1 in all_tools
        assert tool2 in all_tools
    
    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        registry = ToolRegistry()
        registry._tools = {}  # Clear the registry
        
        tool = TestTool(name="test_tool", description="A test tool")
        registry.register_tool(tool)
        
        definitions = registry.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"
        assert definitions[0]["description"] == "A test tool"
        assert "input_schema" in definitions[0]
        assert "output_schema" in definitions[0]