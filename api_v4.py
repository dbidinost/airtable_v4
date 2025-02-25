from fastapi import FastAPI, HTTPException, Response # type: ignore
from datetime import datetime
#from fastapi import JSONResponse, RedirectResponse # type: ignore
from pydantic import BaseModel
from pydantic_settings import BaseSettings # type: ignore
import httpx
import os 
import uvicorn # type: ignore

from langc import v5_assistant_create
from airtable_functions import fetch_airtable_record
from runs import process_run_func
from openai_functions import v5_process_match, v4_process_match

class Settings(BaseSettings):
    airtable_api_key: str
    openai_api_key: str

    class Config:
        env_file = ".env"


settings = Settings()
# airtable_base_id = "apps3g53eD7Wzn7rE"  #Matching_V4
#airtable_table_name = "Companies"
#airtable_base_id = "appf5YIm7Q2CkYiTG" #Matching_V5
app = FastAPI()

@app.get("/")
async def root():
    print("parapp")
    return {"message": "Hello World"}

# Endpoint to create an OpenAI Assistant ID and update Airtable:
@app.post("/create-assistant/{record_id}")
async def create_assistant(record_id: str):
    print(f"Create_Assistant, record:{record_id} at: {datetime.now()}")
    record = await fetch_airtable_record("Companies", record_id, settings)
    # Assuming the file URL is stored under "file_url"
 
    print("Company Record", record['fields'].get('Company Name', 'No Company Name'))
    file_url = record['fields'].get('Attachments', [])
    
    if not file_url:
        raise HTTPException(status_code=404, detail="File URL not found in record")
    
    #assistant_id = "simulated_assistant_id"  # This should be replaced with actual OpenAI API call

    assistant_id = await v5_assistant_create(record, settings)
    return {"assistant_id": assistant_id}

# Endpoint to process matching between Companies and Investors
# record_id is the record ID of the Match record
@app.post("/process-match/{record_id}")
async def use_openai(record_id: str):
    #print("Process match, record:", record_id)
    print(f"Match Started, record:{record_id} at: {datetime.now()}")
    record = await fetch_airtable_record("Matches", record_id, settings)
    match_version = record['fields'].get("Match Version")
    generate_reports_flag = record['fields'].get("Generate Reports")
    steps_per_match = 6 if generate_reports_flag == 'Yes' else 4
    total_steps = 1 * steps_per_match
    
    if match_version.split('.')[0] == '5':    # VERSION 5
        async_result = await v5_process_match(0, total_steps, record, settings,)  #Run ID not passed
    elif match_version.split('.')[0] == '4':    # VERSION 4
        async_result = await v4_process_match(0, total_steps, record, settings,)  #Run ID not passed
    else :
        print(f"Version: {match_version} not supported")
        return None

    if async_result is not None:
        match_score, match_score_text, Token_Acc_Cost = async_result
        return match_score, match_score_text, Token_Acc_Cost
#
# Endpoint to process runs 
# record_id is the record ID of the Match record
@app.post("/process-run/{record_id}")
async def process_run(record_id: str):
    print(f"Run Started, record:{record_id} at: {datetime.now()}")
    record = await fetch_airtable_record("Runs", record_id, settings)
    #record = await fetch_airtable_record("tblTpVdB6sjx2y6iL", record_id, settings)
    async_result = await process_run_func(record, settings)  

    if async_result is not None:
        match_score, match_score_text, Token_Acc_Cost = async_result
        return match_score, match_score_text, Token_Acc_Cost


