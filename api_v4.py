from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, BaseSettings
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import httpx
import os 
from pyairtable import Api

class Settings(BaseSettings):
    airtable_api_key: str
    openai_api_key: str

    class Config:
        env_file = ".env"


settings = Settings()
airtable_base_id = "apps3g53eD7Wzn7rE"
airtable_table_name = "Companies"

app = FastAPI()

# Utility function to get and update Airtable records
async def fetch_airtable_record(record_id: str):
    url = f"https://api.airtable.com/v0/{airtable_base_id}/{airtable_table_name}/{record_id}"
    headers = {"Authorization": f"Bearer {settings.airtable_api_key}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    return response.json()

async def update_airtable_record(record_id: str, field_data: dict):
    url = f"https://api.airtable.com/v0/{settings.airtable_base_id}/{settings.airtable_table_name}/{record_id}"
    headers = {
        "Authorization": f"Bearer {settings.airtable_api_key}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        response = await client.patch(url, json={"fields": field_data}, headers=headers)
    return response.json()

@app.get("/")
async def root():
    print("parapp")
    return {"message": "Hello World"}
# Endpoint to create an OpenAI Assistant ID and update Airtable

@app.post("/create-assistant/{record_id}")
async def create_assistant(record_id: str):
    print("create_assistant, record:", record_id)
    record = await fetch_airtable_record(record_id)
    # Assuming the file URL is stored under "file_url"
    file_url = record["fields"].get("Attachments")
    
    if not file_url:
        raise HTTPException(status_code=404, detail="File URL not found in record")
    
    print("file_url:", file_url)
    # Simulate creating an Assistant ID using file_url
    assistant_id = "simulated_assistant_id"  # This should be replaced with actual OpenAI API call
    
    #await update_airtable_record(record_id, {"assistant_id": assistant_id})
    return {"assistant_id": assistant_id}

# Endpoint to use OpenAI API and update Airtable based on response
@app.post("/use-openai/{record_id}")
async def use_openai(record_id: str, query: str):
    record = await fetch_airtable_record(record_id)
    # Assuming the Assistant ID is stored under "assistant_id"
    assistant_id = record["fields"].get("assistant_id")
    if not assistant_id:
        raise HTTPException(status_code=404, detail="Assistant ID not found in record")

    # Simulate querying OpenAI with assistant_id and query
    response_text = "simulated_response"  # This should be replaced with actual OpenAI API call
    
    await update_airtable_record(record_id, {"response_field": response_text})
    return {"response_text": response_text}

