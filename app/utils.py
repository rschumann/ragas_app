import os
import shutil
import aiohttp
from fastapi import HTTPException

def clear_ragas_cache():
    ragas_cache_dir = os.path.expanduser("~/.cache/ragas")
    if os.path.exists(ragas_cache_dir):
        shutil.rmtree(ragas_cache_dir)

async def api_rag(question: str):
    API_URL = os.getenv('API_URL', 'http://localhost:8000/query')
    PROJECT = os.getenv('PROJECT', 'testing')
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    data = {"project": PROJECT, "question": question}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, json=data, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                return result["final_answer"], [result["local_answer"], result["global_answer"]]
            else:
                raise HTTPException(status_code=response.status, detail="RAG API request failed")
