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
    
    """Convert python function arguemnt schema to json schema, as required by openapi"""
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
    """
    Python function argument wrapper.

    Attributes:
        type (Optional[str]): The type of the argument.
        description (Optional[str]): The description of the argument.
        enum (Optional[list]): The list of possible values for the argument.
    """
    type: Optional[str] = None
    description: Optional[str]
    enum: Optional[list]

    class Config:
        extra = "allow"
    

class Parameters(BaseModel):
    """
    Represents the parameters of a tool.

    Attributes:
        type (str): The type of the parameters.
        required (Optional[list]): The list of required parameters.
        properties (Dict[str, FuncArg]): The properties of the parameters.

    Example:
        parameters = Parameters(
            type="example",
            required=["param1", "param2"],
            properties={
                "param1": FuncArg(type=str),
                "param2": FuncArg(type=int)
            }
        )
    """
    type: str
    required: Optional[list]
    properties: Dict[str, FuncArg]

class Function(BaseModel):
    """
    Represents a function.

    Attributes:
        description (Optional[str]): The description of the function.
        name (str): The name of the function.
        parameters (Parameters): The parameters of the function.
    """
    description: Optional[str]
    name: str
    parameters: Parameters

class ToolSpec(BaseModel):
    type:str = None
    function:Function 
    

class FunctionWrapper:

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
    def tool_spec(self) -> dict:
        raise NotImplementedError("tool_spec function not implemented.")
        
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    def call(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        if not isinstance(result, str):
            result = json.dumps(result)
        return result



    
