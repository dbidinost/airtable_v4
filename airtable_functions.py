
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


async def update_airtable_match_record(match_ID, score, score_text, token_cost , settings):
    field_data = {
    'AI Score': score,  # Change 'FieldName' to your actual field name
    'AI Text': score_text,       # Example of updating another field
    'Last run cost': token_cost       # Example of updating another field
    }
    response = await update_airtable_record("Matches", match_ID, field_data, settings)
    return response