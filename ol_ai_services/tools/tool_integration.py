from typing import List, Dict, Any, Optional
from ol_ai_services.tools.base_tool import BaseTool, ToolInput, ToolOutput
from ol_ai_services.tools.tool_registry import ToolRegistry

class ToolIntegration:
    """
    Class for integrating tools with various LLM frameworks like LangChain, LlamaIndex, etc.
    
    This class provides methods to convert tools from our standard format to
    framework-specific formats.
    """
    
    @staticmethod
    def get_tools_for_langchain() -> List[Any]:
        """
        Get all registered tools converted to LangChain format.
        
        Returns:
            A list of tools in LangChain format
        """
        from langchain.tools import Tool
        
        registry = ToolRegistry()
        langchain_tools = []
        
        for tool in registry.get_all_tools():
            # Create a function that calls the tool's run method
            def tool_func(tool_instance=tool, **kwargs):
                import asyncio
                return asyncio.run(tool_instance.run(**kwargs))
            
            # Create a LangChain tool
            langchain_tools.append(
                Tool(
                    name=tool.name,
                    description=tool.description,
                    func=tool_func,
                    args_schema=tool.input_schema
                )
            )
        
        return langchain_tools
    
    @staticmethod
    def get_tools_for_llama_index() -> List[Any]:
        """
        Get all registered tools converted to LlamaIndex format.
        
        Returns:
            A list of tools in LlamaIndex format
        """
        from llama_index.core.tools import FunctionTool
        
        registry = ToolRegistry()
        llama_index_tools = []
        
        for tool in registry.get_all_tools():
            # Create a function that calls the tool's run method
            def tool_func(tool_instance=tool, **kwargs):
                import asyncio
                return asyncio.run(tool_instance.run(**kwargs))
            
            # Create a LlamaIndex tool
            llama_index_tools.append(
                FunctionTool.from_defaults(
                    name=tool.name,
                    description=tool.description,
                    fn=tool_func
                )
            )
        
        return llama_index_tools
    
    @staticmethod
    def get_tools_for_openai_format() -> List[Dict[str, Any]]:
        """
        Get all registered tools converted to OpenAI function calling format.
        
        Returns:
            A list of tools in OpenAI function calling format
        """
        registry = ToolRegistry()
        openai_tools = []
        
        for tool in registry.get_all_tools():
            # Create an OpenAI function definition
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema.schema()
                }
            })
        
        return openai_tools