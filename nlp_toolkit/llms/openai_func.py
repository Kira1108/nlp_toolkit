import json

def get_completion(client, prompt:str, tools_list:list, verbose:bool = False):
    messages = [{"role": "user", "content":prompt }]

    tools = [
        tool.tool_spec for tool in tools_list
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:

        available_functions = {
            tool.name: tool.func for tool in tools_list
        } 
        messages.append(response_message) 
        for tool_call in tool_calls:
            if verbose:
                print(f"Calling Tool: {tool_call.function.name}")
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            current_function = [tool for tool in tools_list if tool.name == function_name][0]
            args = {arg: function_args.get(arg) for arg in current_function.funcargs.keys()}
            function_response = function_to_call(
                **args
            )
            if verbose:
                print(f"Tool Response: {function_response}")
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            ) 
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        ) 
        return second_response