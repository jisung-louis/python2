from fastapi import APIRouter, Request, Response
router = APIRouter(prefix='/api')

from service import predictService

@router.post('/train')
async def train(request : Request) :    # Request 객체
    data = await request.json()    # request 객체 body값을 직접 json으로 변환.

    return predictService.trainModel(data)

@router.post('/predict')
async def predict(variables: dict):
    return predictService.predict(variables)