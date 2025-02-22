
import httpx
# from pydantic import BaseModel
#from pydantic import BaseSettings
from pydantic_settings import BaseSettings # type: ignore


class Settings(BaseSettings):
    airtable_api_key: str
    openai_api_key: str

    class Config:
        env_file = ".env"


# #settings = Settings()
#airtable_base_id = "apps3g53eD7Wzn7rE"
airtable_base_id = "appf5YIm7Q2CkYiTG" #Matching_V5
###########
# Utility functions to get and update Airtable records
###########
async def fetch_airtable_record(airtable_table_name:str , record_id: str, settings: Settings):
    url = f"https://api.airtable.com/v0/{airtable_base_id}/{airtable_table_name}/{record_id}"
    
    #print("fetch_airtable_record, url:", url)
    #print("key: ", settings.airtable_api_key)
    #print(url)
    headers = {"Authorization": f"Bearer {settings.airtable_api_key}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    return response.json()


async def update_airtable_record(airtable_table_name, record_id: str, field_data: dict, settings: Settings):
    url = f"https://api.airtable.com/v0/{airtable_base_id}/{airtable_table_name}/{record_id}" 
    print("update_airtable_record, url:", url)
    headers = {
        "Authorization": f"Bearer {settings.airtable_api_key}", # type: ignore
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        response = await client.patch(url, json={"fields": field_data}, headers=headers)
    
    
    if response.status_code == 200:
        #print("Record updated successfully:", response.json())
        return True
    else:
        print("Failed to update Airtable record:", response.json())
        return False

###
# Update a MATCH record with final score
##
async def update_airtable_match_record(match_ID, score, score_text, token_cost , settings):
    field_data = {
    'AI Score': score,  # Change 'FieldName' to your actual field name
    'AI Text': score_text,       # Example of updating another field
    'Last run cost': token_cost       # Example of updating another field
    }
    response = await update_airtable_record("Matches", match_ID, field_data, settings)
    return response

###
# Update a match and run record with intermediate step
##
async def update_airtable_step_record(run_ID, match_ID, index, step, total_steps , settings):
    field_data = {
        'AI Text': f"Step {step} of 4..."  # Change 'FieldName' to your actual field name
    }
    response = await update_airtable_record("Matches", match_ID, field_data, settings)

    progress = f"Step {index*4+step} / {total_steps}"
    field_data = {
        'Progress': progress  # Change 'FieldName' to your actual field name
            #   'AI Text': score_text,       # Example of updating another field
            #   'Last run cost': token_cost       # Example of updating another field
        }
    run_air = await update_airtable_record("Runs", run_ID, field_data, settings)
    
######
# Set complete status for a run
######
async def update_airtable_run_complete(run_id, error, settings):
    if error:
        field_data = {
            'Status': 'Error',  
            'Progress': error  
        }        
    else:
        field_data = {
        'Status': 'Done',  
        'Progress': 'Complete'  
    }
    response = await update_airtable_record("Runs", run_id, field_data, settings)
    return response
######
# Delete a record
######

async def delete_airtable_record(airtable_table_name: str, record_id: str, settings: Settings):
    
    url = f"https://api.airtable.com/v0/{airtable_base_id}/{airtable_table_name}/{record_id}"
    
    headers = {"Authorization": f"Bearer {settings.airtable_api_key}"}
    
    print("delete_airtable_record, url:", url)
    
    async with httpx.AsyncClient() as client:
        response = await client.delete(url, headers=headers)

    # Check if the deletion was successful
    if response.status_code == 200:
        print("Record deleted successfully.")
    else:
        print(f"Failed to delete record. Status code: {response.status_code}, Response: {response.text}")
    
    return response.json()
