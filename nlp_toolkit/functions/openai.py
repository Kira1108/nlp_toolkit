from __future__ import annotations
import inspect
from pydantic import BaseModel
from typing import Dict, Optional
import json


"""
openai function calling API
https://platform.openai.com/docs/guides/function-calling

openai function requires a tool parameters, which is in fact a tool specification.
the format of the specification is defined in the link above.

The following code is a schema of the tool specification.
"""

def get_function_arg_type(func) -> dict:
    
    JSON_SCHEMA_TYPE_MAP = {
        dict:  "object",
        list:  "array",
        bool:  "boolean",
        int:   "number",
        float: "number",
        str:   "string"
    }
    
    JSON_SCHEMA_TYPE_MAP_STR = {
        "dict":"object",
        "list":"array",
        "bool":"boolean",
        "int":"number",
        "float":"number",
        "str":"string"
    }

    parsed_args = {name:type.annotation 
                   for name, type  in inspect.signature(func).parameters.items()}

    json_types = {
        arg: JSON_SCHEMA_TYPE_MAP[arg_type] 
        if arg_type in JSON_SCHEMA_TYPE_MAP.keys() 
        else JSON_SCHEMA_TYPE_MAP_STR[arg_type]
    for arg, arg_type in parsed_args.items()
    }

    return json_types

class FuncArg(BaseModel):
    type:Optional[str] = None
    description:Optional[str]
    enum:Optional[list]
    
class Parameters(BaseModel):
    type:str
    required:Optional[list]
    properties:Dict[str, FuncArg]
    
class Function(BaseModel):
    description:Optional[str]
    name:str
    parameters:Parameters

class ToolSpec(BaseModel):
    type:str
    function:Function 
    
class OpenAITool:
    """
    A class representing an OpenAI tool.

    This class provides a convenient way to wrap a function as an OpenAI tool,
    allowing easy access to the function's name, description, and parameters.

    Attributes:
        func: The function to be wrapped as an OpenAI tool.
        funcargs: A dictionary containing the arguments for the function.
    
    Examples:
        # Create an OpenAI tool for a function named 'add_numbers'
        tool = OpenAITool(add_numbers, arg1=FuncArg(type=int), arg2=FuncArg(type=int))

        # Get the name of the function
        print(tool.name)  # Output: 'add_numbers'

        # Get the description of the function
        print(tool.function_description)  # Output: 'A function to add two numbers'

        # Get the required arguments of the function
        print(tool.required)  # Output: ['arg1', 'arg2']

        # Get the tool specification
        print(tool.tool_spec)  # Output: dictionary required by openai
    """

    def __init__(self, func, **kwargs:FuncArg):
        self.func = func
        self.funcargs = kwargs
        
    @property
    def name(self):
        return self.func.__name__
        
    @property
    def function_name(self):
        return self.func.__name__
    
    @property
    def function_description(self):
        return self.func.__doc__
    
    @property
    def required(self):
        return [
            name for name, p in inspect.signature(self.func).parameters.items() 
            if p.default is inspect._empty]
    
    @property
    def tool_spec(self):
        arg_types = get_function_arg_type(self.func)
        
        funcargs = {}
        for k,v in self.funcargs.items():
            # check if input type is not specified
            # and can be retrieved from signature
            if v.type is None and k in arg_types:
                v.type = arg_types[k]
                funcargs[k] = v
            else:
                funcargs[k] = v      
            
        parameters = Parameters(
                    type='object',
                    properties=funcargs)
        
        if len(self.required) > 0:
            parameters.required = self.required
        
        return ToolSpec(
            type="function",
            function=Function(
                name=self.function_name,
                description=self.function_description,
                parameters=parameters
            )
        ).dict(exclude_unset=True)
        
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    def call(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        if not isinstance(result, str):
            result = json.dumps(result)
        return result
