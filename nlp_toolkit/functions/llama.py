from .base import FunctionWrapper
from .base import get_function_arg_type
from .base import Parameters
from .base import FuncArg
from .base import Function

class LlamaFunction(FunctionWrapper):
    
    def __init__(self, func, **kwargs:FuncArg):
        super().__init__(func, **kwargs)
        
    @property
    def tool_spec(self) -> dict:
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

        return Function(
                name=self.function_name,
                description=self.function_description,
                parameters=parameters
            ).dict(exclude_unset=True)