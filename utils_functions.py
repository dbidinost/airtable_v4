import openai
#############
#
# Thread Create
#
#############
async def v4_thread_create(settings):
    client = openai.OpenAI()
    try:
        response = client.beta.threads.create(
             messages=[{
                "role": "user",
                "content": "You will retrieve information from your file, and provide factual and synthetic answers to the questions asked",
                #Uncomment and modify the following line as necessary when including file attachments
                #"attachments": [{"file_id": attachment_field_id, "tools": [{"type": "file_search"}]}]
            }]
        )
        print("API Response:", response)
        print("Thread ID:", response.id)
        return response.id

    except openai.OpenAIError as e:
        #print("Create Thread Error:", response.status_code, response.text)
        print("Create Thread Error:", str(e))
        return None

#############
#
# Cost of Tokens Calculation
#
#############
def calculate_cost_tokens(tokens, model):
    pricing = {
            'gpt-4o': {'input': 2.50, 'output': 10.00},  # per Mtokens
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},    # per Mtokens
            'gpt-4-turbo': {'input': 10.00, 'output': 30.00},  # per Mtokens
            'gpt-4': {'input': 30.00, 'output': 60.00},       # per Mtokens
            'gpt-3.5-turbo': {'input': 1.50, 'output': 2.00}       # per Mtokens: currently point to gpt-3.5-turbo-0613
            }
    # Check if input is an instance of expected Usage class or a dict
    if isinstance(tokens, dict):
        prompt_tokens = tokens['prompt_tokens']
        completion_tokens = tokens['completion_tokens']
    else:
        # Assuming tokens_or_usage is an instance of the Usage class
        prompt_tokens = tokens.prompt_tokens
        completion_tokens = tokens.completion_tokens

    # Determine the pricing structure for the model
    model_pricing = pricing.get(model, {'input': 0.01, 'output': 0.01})  # Default pricing

    # Calculate costs separately for input and output tokens
    input_cost = prompt_tokens * model_pricing['input'] * 0.000001   #Mtokens
    output_cost = completion_tokens * model_pricing['output'] * 0.000001   #Mtokens
    total_cost = input_cost + output_cost  # Optionally recalculated to verify

    return total_cost
