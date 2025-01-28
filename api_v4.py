from fastapi import FastAPI, HTTPException, Response # type: ignore

#from fastapi import JSONResponse, RedirectResponse # type: ignore
from pydantic import BaseModel
from pydantic_settings import BaseSettings # type: ignore
import httpx
import os 
import uvicorn # type: ignore

from openai_functions import v4_assistant_create, v4_process_match
from airtable_functions import fetch_airtable_record

class Settings(BaseSettings):
    airtable_api_key: str
    openai_api_key: str

    class Config:
        env_file = ".env"


settings = Settings()
airtable_base_id = "apps3g53eD7Wzn7rE"
#airtable_table_name = "Companies"

app = FastAPI()

@app.get("/")
async def root():
    print("parapp")
    return {"message": "Hello World"}

# Endpoint to create an OpenAI Assistant ID and update Airtable:
@app.post("/create-assistant/{record_id}")
async def create_assistant(record_id: str):
    print("create_assistant, record:", record_id)
    record = await fetch_airtable_record("Companies", record_id, settings)
    # Assuming the file URL is stored under "file_url"
    
    print("Company Record", record)
    file_url = record['fields'].get('Attachments', [])
    
    if not file_url:
        raise HTTPException(status_code=404, detail="File URL not found in record")
    
    #assistant_id = "simulated_assistant_id"  # This should be replaced with actual OpenAI API call

    assistant_id = await v4_assistant_create(record, settings)
    return {"assistant_id": assistant_id}

# Endpoint to process matching between Companies and Investors
# record_id is the record ID of the Match record
@app.post("/process-match/{record_id}")
async def use_openai(record_id: str):
    print("Process match, record:", record_id)
    record = await fetch_airtable_record("Matches", record_id, settings)

    async_result = await v4_process_match(record, settings)

    if async_result is not None:
        match_score, match_score_text, Token_Acc_Cost = async_result
        return match_score, match_score_text, Token_Acc_Cost


