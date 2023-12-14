from pydantic import BaseModel
from openai import OpenAI
import json
from nlp_toolkit.llms.openai import OpenAIRequest

class PydanticTool:

    def __init__(self, model:BaseModel):

        if not issubclass(model, BaseModel):
            raise ValueError("Model should be a valid pydantic BaseModel")
        self.model = model

    @property
    def tool_spec(self):
        schema = self.model.schema()
        return {
            "type":"function",
            "function":{
                "name":schema['title'],
                "description":schema['description'],
                "parameters": {
                    "type": "object",
                    "properties":{k: {kk:vv for kk, vv in v.items() if kk!= 'title'} 
                                for k, v in schema['properties'].items()}
                }
            }
        }

    def __call__(self, **kwargs):
        return self.model(**kwargs)

class StructuredExtraction:

    def __init__(self, client:"OpenAI", response_model:BaseModel, system_prompt:str = None, **kwargs):
        self.client = client
        self.tool = PydanticTool(response_model)
        if system_prompt is None:
            self.system_prompt = 'you are a nlp bot, that helps to extract entities from texts'
        else:
            self.system_prompt = system_prompt  

    def __call__(self, input):

        openai_request = OpenAIRequest(
            client = self.client,
            model = "gpt-3.5-turbo-0613",
            messages = [{'role':'system','content':self.system_prompt}],
            tools = [self.tool.tool_spec],
            tool_choice = 'auto'
        )

        response = openai_request(input)
        args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        return self.tool(**args)