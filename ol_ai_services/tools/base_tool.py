from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from pydantic import BaseModel, Field

class ToolInput(BaseModel):
    """Base class for tool inputs."""
    pass

class ToolOutput(BaseModel):
    """Base class for tool outputs."""
    success: bool = Field(default=True, description="Whether the tool execution was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if the tool execution failed")

class BaseTool(ABC):
    """
    Base class for all tools that can be used by LLMs.
    
    Tools provide a standardized interface for LLMs to interact with external systems
    and perform specific actions like database queries, API calls, etc.
    """
    
    name: str
    description: str
    input_schema: Type[ToolInput]
    output_schema: Type[ToolOutput]
    
    def __init__(self, name: str, description: str):
        """
        Initialize a new tool.
        
        Args:
            name: The name of the tool, used to identify it in the LLM context
            description: A description of what the tool does, used by the LLM to decide when to use it
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    async def _run(self, input_data: ToolInput) -> ToolOutput:
        """
        Internal method that implements the tool's functionality.
        
        Args:
            input_data: The input data for the tool, validated against the input_schema
            
        Returns:
            The output of the tool execution
        """
        pass
    
    async def run(self, **kwargs) -> ToolOutput:
        """
        Execute the tool with the given inputs.
        
        Args:
            **kwargs: The input parameters for the tool
            
        Returns:
            The output of the tool execution
        """
        try:
            # Validate the input data using the input schema
            input_data = self.input_schema(**kwargs)
            
            # Run the tool implementation
            return await self._run(input_data)
        except Exception as e:
            # Create a failed output with the error message
            return self.output_schema(
                success=False,
                error_message=str(e)
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the tool definition as a dictionary, used for tool registration with LLMs.
        
        Returns:
            A dictionary representation of the tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.schema(),
            "output_schema": self.output_schema.schema()
        }