import httpx, asyncio
import openai
from airtable_functions import fetch_airtable_record, update_airtable_record, delete_airtable_record, update_airtable_run_complete
from openai_functions import v5_process_match, v4_process_match
import time
import datetime

fast_api_url = "https://nltq7knj-8005.uks1.devtunnels.ms" #This is my FastAPI URL
######
#
# Process Run 
# v4 and v5
#######
async def process_run_func(run_record, settings):

    run_id = run_record["id"]
    companies = run_record["fields"].get("Companies") # get returns an array
    investor_ID = run_record["fields"].get("Investor")[0]  # get returns an array
    matches = run_record["fields"].get("Matches")  # get returns an array
    match_version = run_record['fields'].get("Run Version") 
    generate_reports_flag = run_record['fields'].get("Generate Reports")
    print(f"generate_reports_flag: {generate_reports_flag}")
    print(f"Version: {match_version}")

    print(f"Companies: {companies}")
    print(f"Investor ID: {investor_ID}")
    print(f"Matches: {matches}")

    steps_per_match = 6 if generate_reports_flag == 'Yes' else 4
    total_steps = len(matches) * steps_per_match

    for index, match in enumerate(matches):
        match_record = await fetch_airtable_record("Matches", match, settings)
        #print(f"Match Started, record:{match_record["fields"].get("Company")} at: {datetime.datetime.now()}")
        
        #try:   # If current match fails, continue with the next match
        if match_version.split('.')[0] == '5':    # VERSION 5
                async_result = await v5_process_match(index, total_steps, match_record, settings, run_id)
        elif match_version.split('.')[0] == '4':    # VERSION 4
                async_result = await v4_process_match(index, total_steps, match_record, settings, run_id)
        else :
                print(f"Version: {match_version} not supported")
                return None
        match_score, match_score_text, Token_Acc_Cost = async_result
        print(f"=====MATCH COMPLETED, record_id:{match}, version: {match_version}  at: {datetime.datetime.now()}")
        #print(f"Async_result: {async_result}")

        #except Exception as e:
         #   print(f"Match Failed: {str(e)}, record_id:{match} at: {datetime.datetime.now()}")
        

        # except Exception as e:
        #     print(f"Match Failed: {str(e)}, record_id:{match} at: {datetime.datetime.now()}")
        #     await match_cleanup(match, settings)
        
        # if async_result:
        #     match_score, match_score_text, Token_Acc_Cost = async_result
        #     print(f"Match Completed, record_id:{match} at: {datetime.datetime.now()}")
        # else:
        #     await match_cleanup(match, settings)
        #     print(f"Match Failed, record:{match} at: {datetime.datetime.now()}")
    

    response = await update_airtable_run_complete(run_id,'',settings)  # second parameter is error must be empty
    return None
    
    #time.sleep(5)


# OLD ATTEMPT AT CONCURRENCY:

    # Process each match in the run:
#     for match in matches:
#         send_to_myself(match) 
#         time.sleep(5)

# #####
# #
# # API call to FastAPI for match processing
# #
# #####
# async def send_to_myself(match_id):
    
#     fetch_address = f"{fast_api_url}/process-match/{match_id}"
#     print(f"Send to myself {fetch_address}")
#     # Making a POST request
#     async with httpx.AsyncClient() as client:
#         response = await client.post(fetch_address, timeout=120.0  )
    
#     if response: 
#         print(response.status_code)
#         print(response.text)


# # # Running the async tasks
# #     results = asyncio.run(run_all(matches,settings))
# #     #v5_process_match(record, settings):
# #     print(f"Asyncio Result: {results}")
# #     return None

# # async def run_all(matches, settings):
# #     tasks = [v5_process_match(match) for match in matches]
# #     results = await asyncio.gather(*tasks)
# #     return results

########
# Cleanup of matches not completed:
########
async def match_cleanup(match_id, settings):
    # Construct the URL for the DELETE request
    resp = await delete_airtable_record("Matches", match_id, settings)
    return resp

