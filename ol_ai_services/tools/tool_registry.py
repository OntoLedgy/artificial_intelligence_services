from typing import Dict, List, Type, Optional
from ol_ai_services.tools.base_tool import BaseTool

class ToolRegistry:
    """
    Registry for tools that can be used by LLMs.
    
    Provides a central registry for registering and retrieving tools.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            cls._instance._tools = {}
        return cls._instance
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool with the registry.
        
        Args:
            tool: The tool to register
        """
        self._tools[tool.name] = tool
    
    def register_tools(self, tools: List[BaseTool]) -> None:
        """
        Register multiple tools with the registry.
        
        Args:
            tools: The tools to register
        """
        for tool in tools:
            self.register_tool(tool)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: The name of the tool to get
            
        Returns:
            The tool if found, None otherwise
        """
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all registered tools.
        
        Returns:
            A list of all registered tools
        """
        return list(self._tools.values())
    
    def get_tool_definitions(self) -> List[Dict]:
        """
        Get the definitions of all registered tools.
        
        Returns:
            A list of tool definitions
        """
        return [tool.to_dict() for tool in self._tools.values()]