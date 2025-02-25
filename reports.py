######################
import openai
import httpx

from pydantic_settings import BaseSettings
from airtable_functions import update_airtable_record, fetch_airtable_record
from utils_functions import calculate_cost_tokens, v4_thread_create
######################
# Function to generate investor and entrepreneur reports
# V4
######################
async def v4_generate_reports(assistant_id, match_record, settings): 
     
    thread_id = await v4_thread_create(settings) # No cost in creating Threads.
    if not thread_id:
        print("Thread Creation Failed")
        return None
    model = "gpt-4o-mini"
    completion_answer, cost = await assistant_generate_report(thread_id, assistant_id, match_record, model, settings)
    
    return completion_answer, cost

########
# Add message to Assistant Thread
# Get criteria text in input 
# V4 only
########
async def assistant_generate_report(thread_id, assistant_id, match_record, model, settings):
    # Collect info of the 'Match' field
    #company_ID = match_record["fields"].get("Company")[0] # get returns an array
    investor_name = match_record["fields"].get("InvestorName (from InvestorName)")[0]  # get returns an array
    company_name = match_record["fields"].get("CompanyName (from Companies)")[0]  #Lookout field 
    criteria_text = match_record["fields"].get("AI Text")  # match object
    ai_score = match_record["fields"].get("AI Score")  # match object
    #match_ID = match_record['fields'].get("Match ID")  # match object
    print(f"Generate Report for: {investor_name}; {company_name}")

    async def openai_assistant_report(user_question):
        # Add user message to the thread
        try:
            message_response = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_question,
            )
            
        except openai.APIError as e:
        #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            return None
        except openai.APIConnectionError as e:
        #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            return None
        except openai.RateLimitError as e:
        #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            return None
        
        #print("Added user message to the thread", message_response)

        run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id,
        model=model
            )
        
        print("Run create and poll with status: ", run.status)
    
        cost = calculate_cost_tokens(run.usage, model)
        if run.status == "completed":
            #print("===Run completed with run: ", run)
            #print("===Cost: ", cost)
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            #print("Run completed with status: " + run.status)
            #print("messages: ", messages)
            for message in messages:
                assert message.content[0].type == "text"
                #print({"role": message.role, "message": message.content[0].text.value})
                return message.content[0].text.value, cost
        else:
            if run.status == "failed":   
                print("===Run failed with run: ", run)
                print("===Cost: ", cost)
                return "Failed: treat as null question", cost
            else:
                if run.status == "incomplete":
                    print("===Run incomplete with run: ", run)
                    print("===Cost: ", cost)
                    return "Failed: treat as null question", cost
    
    client = openai.OpenAI()
    user_inv_question = f"""You write reports for investor (<investor>) investment committe  about assessment of the company <company>
        The section below <criteria_text> is a list of json objects with: criteria name, investor preference, 
        company response, and the match score.
        
        Please write a one-page report in md format with following information/sections:
        1) Company name, General company information from <company>
        2) Investor name from: <investor>
        3) Analyze <criteria_text> and describe the 3-4 best (+2) and worst (-2) performing criteria and describe them as strengths and weaknesses of the company. 
        4) Provide final score (from <ai_score>) and provide final considerations about the investment decision.
                
        Output the answer as markdown text.

        <company>:{company_name} is the name of the company
        <criteria_text>:{criteria_text} 
        <investor>:{investor_name} is the name of the investor
        <ai_score>:{ai_score} is the score of the match

        """
    print("Starting assistant_generate investor...")
    
    md_text, cost = await openai_assistant_report(user_inv_question)    
    
    # Step 1: Save md_text to a file (e.g., document.md)
    file_name = f"docs/Reports/inv_report{company_name}.md"
    with open(file_name, "w") as f:
        f.write(md_text)

    print("File written: ", file_name)

    async with httpx.AsyncClient() as client:
        # Step 2: Upload the file to transfer.sh to get a publicly accessible URL
        upload_url = f"https://transfer.sh/{file_name}"
        with open(file_name, "rb") as f:
            file_content = f.read()
        response = await client.put(upload_url, content=file_content)
        if response.status_code != 200:
            raise Exception("File upload failed")
        file_url = response.text.strip()
        print("File available at:", file_url)
        
        # Step 3: Update the Airtable record with the attachment URL
        attachment_field = "Investor_Report"  # Change if your field name is different

        data = {
            "fields": {
                attachment_field: [
                    {"url": file_url}
                ]
            }
        }
        
        update_airtable_record("Matches", match_record, data, settings)
        return md_text, cost
        
    # Add user message to the thread
