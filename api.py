import nest_asyncio
from fastapi import FastAPI, HTTPException
from app.config import setup_environment, setup_logging
from app.models import EvaluationRequest, EvaluationResponse
from app.services import evaluate_rag
import uvicorn

# Apply nest_asyncio to allow re-entering the event loop
nest_asyncio.apply()

# Setup environment and logging
setup_environment()
logger = setup_logging()

app = FastAPI()

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag_endpoint(request: EvaluationRequest):
    try:
        return await evaluate_rag(request)
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
