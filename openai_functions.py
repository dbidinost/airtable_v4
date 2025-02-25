import httpx, asyncio
import openai
from airtable_functions import fetch_airtable_record, update_airtable_record, update_airtable_match_record,\
    update_airtable_step_record, update_airtable_run_complete
import json
import re
import datetime
import os

from langc import v5_retrieve_criteria_from_company, v5_assistant_create  # my langchain.py
from reports import v4_generate_reports
from utils_functions import calculate_cost_tokens, v4_thread_create
#from langchain_openai import ChatOpenAI




#airtable_base_id = "apps3g53eD7Wzn7rE"  V4
#airtable_base_id = "appf5YIm7Q2CkYiTG"
#############
# 
# ASSISTANT CREATE
# Returns None if failed, or Assistant_ID if successful
#############  
async def v4_assistant_create(company_record, settings):
    # Collect info of the fields
    attachments = company_record["fields"].get("Attachments")
    one_line_pitch = company_record["fields"].get("One Line Pitch")
    company_name = company_record["fields"].get("Company Name")
    solution = company_record["fields"].get("What is your company's solution to this problem?")
    problem = company_record["fields"].get("What is the problem that your company is addressing?")
    one_line_pitch = company_record["fields"].get("One Line Pitch")
    ## Currently only one file is supported: 
    if attachments and len(attachments) > 0:
        pdf_url = attachments[0]['url']
        filename = attachments[0]['filename']
        print(f"pdfUrl: {pdf_url}")
    else:
        print("Could not find Attachment")
        return None
    ###
    # 1. UPLOAD FILE IN OPENAI:
    ###
    async with httpx.AsyncClient() as client:
        response_file = await client.get(pdf_url, timeout=45)
        file_content = response_file.content

        # Create a multipart form data object
        files = {'file': (filename, file_content, 'application/pdf')}
        data = {'purpose': 'assistants'}
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "OpenAI-Beta": "assistants=v2"
        }
        response = await client.post("https://api.openai.com/v1/files",
                                     files=files, data=data, headers=headers, timeout=30)

        if not response.is_success:
            print("Failed File Error:", response.status_code, response.reason_phrase)
            print("File Response Body:", response.text)
            return None

        file_response_data = response.json()
        print("File Upload Response:", file_response_data)
        attachment_file_id = file_response_data['id']

    ###
    # 2. CREATE VECTOR STORE
    ###
    print("Creating Vector Store...")
    vs_name = f"vs4_{company_name}"
    vector_store_payload = {
        "name": vs_name,
        "file_ids": [attachment_file_id]
    }
    headers.update({"Content-Type": "application/json"})
    async with httpx.AsyncClient() as client:
        vectore_store = await client.post("https://api.openai.com/v1/vector_stores",
                                          json=vector_store_payload, headers=headers, timeout=45.0)

        if vectore_store.is_success:
            vs_res = vectore_store.json()
            print("VS Created Successfully:")
            print("VS ID:", vs_res['id'])
            print("VS File Counts:", vs_res['file_counts']['total'])
        else:
            print("Failed to create vector store:", vectore_store.text)
            return None

        if await wait_vector_store_completion(vs_res['id'], settings) != "completed":
            return None

    ###
    # 3. CREATE ASSISTANT
    ###
    print("Creating Assistant...")
    assistant_name = f"v4_{company_name}_-AI_Invest_Assist"
    assistant_payload = {
        "name": assistant_name,
        "instructions": f"""
                    You have access to company investment pitch as a file. You answer questions about\
                    the company and the investment proposition. \
                    You provide answers based both on your provided files and general LLM knowledge. \
                    You will also consider the following additional information provided by the entrepreneur:\
                    <One line pitch> : {one_line_pitch}; <Problem to bo solved>: {problem}; <company's solution>: {solution}
                    """,
        "tools": [{"type": "file_search"}],
        "tool_resources": {"file_search": {"vector_store_ids": [vs_res['id']]}},
        "model": "gpt-4o"
        }
    #timeout = httpx.Timeout(15.0, connect=60.0)
    async with httpx.AsyncClient() as client:
        assistant = await client.post("https://api.openai.com/v1/assistants",
                                      json=assistant_payload, headers=headers, timeout=45.0)

        if assistant.is_success:
            assist_response_data = assistant.json()
            print("Assistant Created Successfully:")
            #print("Assistant:", assist_response_data)
            
            return assist_response_data['id']
        else:
            print("Failed to create assistant:")
            print("Status:", assistant.status_code)
            print("Message:", assistant.text)
            return None

#############
# 
# AI Matching between Company and Investor
# Returns None if failed, or response_match if successful
#############  
async def v4_process_match(index, total_steps, record, settings, run_id=None):
    # Function to update step:
    async def airtable_update_step(step_number):
        match_step_text = f"Step {step_number} / {steps_per_match}..."
        run_step_text = f"Step {index*steps_per_match+step_number}/{total_steps}..."
        response = await update_airtable_step_record(run_id, match_ID, match_step_text, run_step_text , settings)
    
    
    Token_Acc_Cost = 0
    # FETCH ALL AIRTABLE RECORDS
    #
    # Collect info of the 'Match' fields
    company_ID = record["fields"].get("Company")[0] # get returns an array
    investor_ID = record["fields"].get("Investor")[0]  # get returns an array
    #company_name = record["fields"].get("CompanyName (from Companies)")[0]  #Lookout field 
    match_ID = record['fields'].get("Match ID")  # match object
    generate_reports_flag = record['fields'].get("Generate Reports")
    steps_per_match = 6 if generate_reports_flag == 'Yes' else 4

    print(f"==== START RUN AT:, {datetime.datetime.now()}, {generate_reports_flag}")
    #
    # # Collect info of the 'Companies' fields
    # company_url = 'https://api.airtable.com/v0/airtable_base_id/Companies/' + company_ID  # This is inside the fetch_airtable_record function
    company_resp = await fetch_airtable_record("Companies", company_ID, settings)
    #print("\nresp:", company_resp, "\n")
    try:
        assistant_id = company_resp['fields']['AI-AssistantID']
    except KeyError as e:
        assistant_id = None  
    
    
    if not assistant_id:
        print("Assistant ID not found in company record")
        print("Starting Assistant Creation first...")
        assistant_id = await v4_assistant_create(company_resp, settings)  # Create Assistant
        print("Assistant ID:", assistant_id)

    if not assistant_id:  #second time around: failed to create assistant
        print("Assistant ID could not be created")
        response = await update_airtable_match_record(match_ID, 'n/a', 'Assistant not created', '0' , settings)
        return None
    else:
        # In this case, save the assistant ID back to the company record
        #CompanyID = record['fields'].get("RecordID")
        response = await update_airtable_record('Companies', company_ID, {'AI-AssistantID': assistant_id}, settings)
    #
    # Collect info of the 'Companies' fields
    #
    #print("Investor_ID:", investor_ID)
    investor = await fetch_airtable_record("Investors", investor_ID, settings)
    #print("Investor:", investor)
    #investor_criteria_text = investor['fields']['Investor Criteria Text'] 4.0
    investor_must_config = investor['fields'].get('Must Config')
    investor_websites = investor['fields'].get('Invested Companies Websites')
    ######################
    # Algorithm to match the company and investor:
    ######################
    # 0) Wait in Airtable...
    #response = await update_airtable_match_record(match_ID, 'Wait Step1/4...', settings)
    await airtable_update_step(1)
    print("1) Extract criteria from Investor (as list of objects)...")
    criteria_list = await extract_airtable_investor_criteria(investor,settings)
    print(f"1) Criteria List Extracted from Airtable: {criteria_list})")
    
    ######################  
    # 2) Retrieve all criterias response from Assistant (pitch deck)
    print("2) Retrieve all criterias response from Assistant (pitch deck)...")
    await airtable_update_step(2)

    model="gpt-4o-mini"
    updated_criterias_list, retrieval_cost = await retrieve_criteria_from_company(criteria_list, assistant_id,settings, model)
    Token_Acc_Cost += retrieval_cost # cost is calculated in the function
    #print("2) Updated Criterias List:", updated_criterias_list)
    print(f"2) Updated Criterias List Obtained. The retrieval cost is: {retrieval_cost}$)")
   
    if updated_criterias_list == None:
        print("Could not retrieve criteria from company")
        return None 
    ######################  
    # 3) Match the company and investor and rate 
    print("3) Match the company and investor and rate...")
    await airtable_update_step(3)
    model="gpt-4o"
    match_score, match_score_text, acc_cost = await match_company_investor(updated_criterias_list, investor_must_config, investor_websites,settings, model)
    print("Match cost:", acc_cost)
    Token_Acc_Cost += acc_cost  # cost is calculated in the function
    json_list = json.dumps(match_score_text, indent=4, ensure_ascii=False)
    print("Final Match Score:",  match_score)
    #print("FINAL_MATCH_SCORE_TEXT:", json_list)
    await airtable_update_step(4)
    if generate_reports_flag == 'Yes': 
        response, cost = await v4_generate_reports(assistant_id, record, settings)
        Token_Acc_Cost += cost

    print("Final Total Cost:",  Token_Acc_Cost)
    
    #####
    # 4) Update Airtable Fields
    #####
    print("4) Update Airtable Fields to Complete...")
    response = await update_airtable_match_record(match_ID, match_score, json_list, str(round(Token_Acc_Cost,4)) , settings)
    if run_id is not None:
        response = await update_airtable_run_complete(run_id,'',settings)  # second parameter is error must be empty
    #print("Airtable Update Response:", response)
    if not response:
        print("Failed to update Airtable record")
        return match_score, match_score_text, Token_Acc_Cost
    print("Airtable Updated Successfully...")
    print("==== END RUN AT", datetime.datetime.now())
    return match_score, match_score_text, Token_Acc_Cost



####
#
# extract_investor_criteria from Airtable Investor Criteria Text
# output as alist of object  {criteria_name,criteria_text}
####
async def extract_airtable_investor_criteria(investor,settings):

    linked_prefs = investor['fields'].get('Investor Preferences', [])
    #print("Linked Preferences:", linked_prefs)
    preferences_list = []
    for pref_id in linked_prefs:
        # Fetch the specific record by its ID
        pref_record = await fetch_airtable_record("Investor_Pref", pref_id, settings)
        # Extract 'Question', 'Preference', and 'Weight' fields
        question = pref_record['fields'].get('Name (from Question)', '')
        preference = pref_record['fields'].get('Preference', '')
        weight = pref_record['fields'].get('Weight', '')
        
        # Print the details of each preference
        #print(f"pref_id: {pref_id}  Question: {question}")
        #print(f"pref_id: {pref_id}  Preference: {preference}")
        #print(f"pref_id: {pref_id}  Weight: {weight}")

        preferences_list.append({
                'Question': question[0],
                'Preference': preference,
                'Weight': weight
            })

    return preferences_list



####
# Retrieve all criterias response from Assistant (pitch deck)
# V4 Only
####
async def retrieve_criteria_from_company(investor_criterias_list, assistant_id,settings, model):

    
    thread_id = await v4_thread_create(settings) # No cost in creating Threads.
    if not thread_id:
        print("Thread Creation Failed")
        return None
    
    updated_criterias_list = []
    retrieve_cost = 0
    for criteria in investor_criterias_list:
        # Only extract information if the criteria text is not empty:
        if not criteria['Preference']:
            print("Criteria Text is empty: ", criteria['Question'])
            continue
        completion_answer, cost = await assistant_process_question(criteria['Question'], thread_id, assistant_id, model)
        retrieve_cost += cost
    # Create a new dictionary with the existing data and the new completion_answer
        updated_criteria = {
            "criteria_name": criteria['Question'],
            "investor_preference": criteria['Preference'],
            "weight":  criteria['Weight'],
            "company_retrieve": completion_answer
            }
        # Append the new dictionary to the updated_criterias_list
        updated_criterias_list.append(updated_criteria)

    return updated_criterias_list, retrieve_cost
        # 'Other' fields -
        # if criteria['criteria_name'].startswith("Other"):
        #     criteria['criteria_name']=criteria['criteria_text'] # Rename the criteria name for Other.. fields
######
#
# Match the company and investor and rate
# V4 and V5
######
async def match_company_investor(updated_criterias_list, investor_must_config, investor_websites,settings, model):

    # Calculate the match score
    match_score = 0
    acc_cost = 0
    match_score_list = []
    for criteria in updated_criterias_list:
        score, cost = await match_individual_criteria(criteria["investor_preference"], criteria["company_retrieve"], model)
        acc_cost += cost
        # Check if the company has the investor's must-have configuration
        if investor_must_config == "Hard-Must" and score == -2:
            match_score = -100
            #match_score_text = f"{criteria["criteria_name"]}: Hard-must criteria not met"
            match_score_list.append({"criteria_name": criteria["criteria_name"], "investor_preference": criteria["investor_preference"],\
                                  "company_retrieve": criteria["company_retrieve"], "score": "HARD CRITERIA NOT MET"})            
            break
        if (criteria["weight"])<1: criteria["weight"]=1
        if (criteria["weight"])>5: criteria["weight"]=5
        match_score += int(score)*int(criteria["weight"]) # Added weight in 4.1
        #match_score_text += f"|{criteria["criteria_name"]}|{criteria["company_retrieve"]}|{match_score}|\n"
        match_score_list.append({"criteria_name": criteria["criteria_name"], "investor_preference": criteria["investor_preference"],\
                                  "company_retrieve": criteria["company_retrieve"], "score": score})
    # Return the match score and match score text
    return match_score, match_score_list, acc_cost

######
#
# Match the company and investor and rate
# V4 and V5
######
async def match_individual_criteria(investor_preference, company_retrieve, model):
    print(f"match_individual_criteria: Investor Preference: {investor_preference}; \nCompany Retrieve: {company_retrieve}")
    #print("match_individual_criteria: Company Retrieve:", company_retrieve)
    delimiter = "####"
    system_match_message = f"""
    You are evaluation assistant for an investor in startups.
    You are assess how well, for a given criteria,  a company matches the investor's preferences.
    You will be provided what the investor requires about the criteria: <investor_preference> (delimited by {delimiter}).
    You will also be provided with the company characteristic: <company_retrieve> (delimited by {delimiter}).

    Grade (from -3 to 2) how well the company satisfies the investor preference criteria\
    Score: 

    2: Very High: when the company satisfies perfectly the investor's preference for this criteria. \
        If the investor preference is a range, and the company characteristic is within the range, assign 2.
    1: High Match : when there is some evidence of a match but not strong \
        For example if characteristic is a range, assign '1' if the range is not met by 30%, e.g. 4m (millions) against 2-3m
    0: Neutral Match : there is no evidence of a match or mismatch \
        For example if characteristic is a range, assign '0' if the range is not met by 70%. e.g. 5m (millions) against 2-3m
        If the investor preference does not require a specific criteria, assign 0.
    -1: Low Match : when there is some (but not strong) evidence of a the company not matching the preference  \
        For example if characteristic is a range, assign '-1' if the range is not met by 100%. e.g. 6m (millions) against 2-3m
    -2: Very Low Match : when the company's response does not match at all the required investor preference.  \
        For example if characteristic is a range, assign '-2' if the range is not met by more than  100%. e.g. >6m (millions) \
        or <1m against 2-3m
    -3: Assign -3 only if: 1) the investor preference is a  'must' condition, and the company does not meet the criteria, or \
        2) if the investor preference is a 'must not' condition and the company meets the criteria.

        
    'SEIS/EIS': SEIS is more demanding than EIS, so if the investor preference has SEIS, and the company is EIS, assign -1.
    if the investor preference is EIS, and the company has SEIS or EIS, assign 2.   

    'Business Traction' criteria: Strong business traction is when the company has a lot of users, and they keep using the product. \
        Phases/extent of business traction (from lowest to highest) are:\
        Idea, Prototype, MVP or PoC, Beta, Launched, Scaling, Scaled. 
        If the company is in a higher phase than the investor preference, assign 2.\
        If the company is in a lower phase than the investor preference, assign -2 or -1. If the company is in the same phase, assign 1. \
        If the company is in a higher phase than the investor preference, assign 2. 
         

    'Round size': it is the amount of money that the company is looking (seeking) to raise.
    
    'IP': Strength of IP is determined by: patents (higher), trademarks, copyrights, proprietary trade secrets, nothing in particular (lower).

    'Product Market fit': if the company has a stronger product market fit than investor preference, assign 2. \
        If the company has much weaker product market fit then investor preference, assign -2. Assign -1 , 0, 1 in intermediate cases.
            
    If investor does not require a specific preference criteria, assign 0: if the company nevertheless shows a very positive characteristic, assign 1. 

    Provide the match score as an integer. Only output the match score, with nothing else.
    """
    user_message = f"""
        {delimiter} <investor_preference> : {investor_preference} 
        {delimiter} <company_retrieve> : {company_retrieve}
                """
    messages =  [  
    {'role':'system', 
    'content': system_match_message},    
    {'role':'user', 
    'content': user_message},  
    ] 

    response_content,  token_dict = get_completion_and_token_count(messages, model)
    cost = calculate_cost_tokens(token_dict, model)
    print("match_individual_criteria: Score:", response_content)
    print("\n")
    return response_content, cost
########
# Add message to Assistant Thread
# Get criteria text in input 
# V4 only
########
async def assistant_process_question(criteria_text, thread_id, assistant_id, model):
    client = openai.OpenAI()
    user_question = f"""Please retrieve the following information from your file about the company: {criteria_text}. 
        respond in few sentences in a very syntetic way, respond as if you are stating the facts of the company about
        {criteria_text}.
        Round size is the amount of money that the company is looking (seeking) to raise.
        Pre-money valuation (PMV) is the valuation of the company before the investment.\
        It could be calculated from the round size and the percentage of the company that the investor is buying.\
        
        Output the answer as a valid string, with double quotes.
        """
    print("Starting assistant_process_question...")
    
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

# ////////
# //
# // Wait for Vector Store Completion, in 3 seconds polling 
# // Returns 'completed' after files are processed
# // V4
# ////////
async def wait_vector_store_completion(vs_id, settings):
    max_timeout = 15
    index = 0
    while index < max_timeout:
        async with httpx.AsyncClient() as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.openai_api_key}",
                "OpenAI-Beta": "assistants=v2"
            }
            response = await client.get(f"https://api.openai.com/v1/vector_stores/{vs_id}", headers=headers)
            vs_res = response.json()
            if not response.is_success:
                print("Error Retrieving VectorStore:")
                print("Run Status Body Error:", vs_res)
                return "Failed 7"
        
            loop_control = (vs_res['status'] == "completed")
            print(f"VS files completed: {vs_res['file_counts']['completed']} out of {vs_res['file_counts']['total']}")
            if loop_control:
                break

            index += 1
            print(f"Run Wait: {index * 3} seconds..")
            await asyncio.sleep(3)  # Sleep for 3 seconds

    if index == max_timeout:
        print(f"Error Retrieving Response: Run Timeout")
        return "Failed 7a"

    return "completed"


#############
#
# Completion OPENAI functions
#
#############
def get_completion(prompt, model="gpt-3.5-turbo"):
    client = openai.OpenAI()
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=500):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        max_tokens=max_tokens, # the maximum number of tokens the model can ouptut 
    )
    return response.choices[0].message.content

def get_completion_and_token_count(messages, 
                                   model="gpt-3.5-turbo", 
                                   temperature=0, 
                                   max_tokens=500):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens,
    )
    
    #print("OpenAIResponse:", response)
    content = response.choices[0].message.content  # response is pydantic model
    
    token_dict = {
    'prompt_tokens':response.usage.prompt_tokens,
    'completion_tokens':response.usage.completion_tokens,
    #'total_tokens':response.usage.total_tokens,
    }

    return content,  token_dict



#############
# V5 AI Matching between Company and Investor
# Returns None if failed, or response_match if successful
#############  
async def v5_process_match(index, total_steps, match_record, settings, run_id=None):

    
    Token_Acc_Cost = 0
    
    # FETCH ALL AIRTABLE RECORDS
    #
    # Collect info of the 'Match' field
    company_ID = match_record["fields"].get("Company")[0] # get returns an array
    investor_ID = match_record["fields"].get("Investor")[0]  # get returns an array
    company_name = match_record["fields"].get("CompanyName (from Companies)")[0]  #Lookout field 
    match_ID = match_record['fields'].get("Match ID")  # match object

    #return
    print(f"==== START RUN AT: {datetime.datetime.now()};MATCH ID: {match_ID}; COMPANY: {company_name}")
    #
    # COLLECT VECTOR STORE??:
    company_record = await fetch_airtable_record("Companies", company_ID, settings)
    vector_db = await v5_assistant_create(company_record, settings) 
    if not vector_db:  #second time around: failed to create assistant
        print("vector store could not be created")
        #response = await update_airtable_match_record(match_ID, 'n/a', 'Assistant not created', '0' , settings)

    # END Collect info of the 'Match' field
    
    investor = await fetch_airtable_record("Investors", investor_ID, settings)
    investor_must_config = investor['fields']['Must Config']
    investor_websites = investor['fields'].get('Invested Companies Websites')

    ######################
    # Algorithm to match the company and investor:
    ######################
    # 0) Wait in Airtable...
    #response = await update_airtable_match_record(match_ID, 'Wait Step1/4...', 'Wait Step1/4...', 'Wait Step1/4...' , settings)
    response = await update_airtable_step_record(run_id, match_ID, index, 1, total_steps , settings)
    # 1) Extract criteria from Investor (as list of object)
    print("1) Extract criteria from Investor (as list of objects)...")
    criteria_list = await extract_airtable_investor_criteria(investor,settings)
    print(f"1) Criteria List Extracted from Airtable: {criteria_list})")
    ######################  
    # 2) Retrieve all criterias response from Assistant (pitch deck)
    ######################
    print("2) Retrieve all criterias response from vector store (pitch deck)...")
    response = await update_airtable_step_record(run_id, match_ID, index, 2, total_steps , settings)
    model="gpt-4o-mini"
    
    #updated_criterias_list, retrieval_cost = await v5_retrieve_criteria_from_company(criteria_list, vector_store, settings, model)
    updated_criterias_list, retrieval_cost = await v5_retrieve_criteria_from_company(criteria_list, vector_db, settings, model)
    # 
    Token_Acc_Cost += retrieval_cost # cost is calculated in the function
    #print("2) Updated Criterias List:", updated_criterias_list)
    print(f"2) Updated Criterias List Obtained. The retrieval cost is: {retrieval_cost}$)")
   
    if updated_criterias_list == None:
        print("Could not retrieve criteria from company")
        return None 
    ######################  
    # 3) Match the company and investor and rate 
    print("3) Match the company and investor and rate...")
    #response = await update_airtable_match_record(match_ID, 'Wait Step3/4...', 'Wait Step3/4...', 'Wait Step3/4...' , settings)
    response = await update_airtable_step_record(run_id, match_ID, index, 3, total_steps , settings)
    model="gpt-4o"
    match_score, match_score_text, acc_cost = await match_company_investor(updated_criterias_list, investor_must_config, investor_websites,settings, model)
    print("Match cost:", acc_cost)
    Token_Acc_Cost += acc_cost  # cost is calculated in the function

    json_list = json.dumps(match_score_text, indent=4, ensure_ascii=False)
    print("FINAL_MATCH_SCORE:",  match_score)
    #print("FINAL_MATCH_SCORE_TEXT:", json_list)
    print(f"FINAL_TOTAL_COST: ${Token_Acc_Cost}")
    #######################
    # 4) Update Airtable Fields
    print("4) Update Airtable Fields...")
    response = await update_airtable_match_record(match_ID, match_score, json_list, str(round(Token_Acc_Cost,4)) , settings)
    
    #print("Airtable Update Response:", response)
    if not response:
        print("Failed to update Airtable record")
        return match_score, match_score_text, Token_Acc_Cost
    print("Airtable Updated Successfully...")
    print(f"==== END RUN AT {datetime.datetime.now()}, COMPANY: {company_name}")
    return match_score, match_score_text, Token_Acc_Cost


