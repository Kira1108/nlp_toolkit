from tenacity import retry, wait_random_exponential, stop_after_attempt
from pydantic import BaseModel, Field
from typing import Optional, Any

class OpenAISingleResponse(BaseModel):

    id:Optional[str] = None
    finish_reason:Optional[str] = None
    content:Optional[str] = None
    role: Optional[str] = 'assistant'
    function_call:Optional[str] = None
    tool_calls:Optional[list] = None
    created:Optional[int] = None
    model:Optional[str] =  'gpt-3.5-turbo-1106'
    object:Optional[str] = 'chat.completion'
    system_fingerprint:Optional[str] = None
    completion_tokens:Optional[int] = None
    prompt_tokens:Optional[int] = None
    total_tokens:Optional[int] = None

    class Config:
        extra = 'ignore'

    @classmethod
    def from_openai_response(cls, response):
        response = response.dict()
        return cls(
            id = response["id"],
            created = response["created"],
            model = response["model"],
            object = response["object"],
            system_fingerprint = response["system_fingerprint"],
            **response["choices"][0]['message'],
            **response["usage"]
        )

class OpenAIRequest(BaseModel):
    
    client:Any
    model:str = "gpt-3.5-turbo-1106"
    messages:list = Field(default_factory= list)
    frequency_penalty:Optional[float]
    logit_bias:Optional[dict]
    max_tokens:Optional[int]
    n:Optional[int]
    presence_penalty:Optional[float]
    response_format:Optional[Any]
    seed:Optional[int]
    stop:Optional[Any]
    stream:Optional[bool]
    temperature:Optional[float]
    top_p:Optional[float]
    tools:Optional[list]
    tool_choice:Optional[Any]
    user:Optional[str]
    
    class Config:
        extra = 'allow'
    
    @property
    def call_args(self):
        args = self.dict(exclude_unset = True)
        return {k:v for k, v in args.items() 
            if k not in ['client','messages']}

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def call_llm(self, prompt:str)-> OpenAISingleResponse:
        self.messages.append({"role": "user","content": prompt})
        response = self.client.chat.completions.create(
            messages = self.messages,
            **self.call_args)    
        return response
        # return OpenAISingleResponse.from_openai_response(response)

    def __call__(self, prompt:str) -> OpenAISingleResponse:
        return self.call_llm(prompt)