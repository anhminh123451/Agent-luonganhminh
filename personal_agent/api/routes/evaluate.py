from fastapi import APIRouter, Depends, Request, Response, status, HTTPException
from evaluate.latency import EvaluateEndToEndLatency
from api.schemas import ChatRequest

router = APIRouter(tags=["Evaluate"])

@router.post("/evaluate")
async def evaluate_e2e_latency(
    request:ChatRequest
):  
    try:
        e2eTime, totalStep = EvaluateEndToEndLatency(request)
        return {
            "e2eTime": e2eTime,
            "totalStep": totalStep,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

