from fastapi import FastAPI
import uvicorn

app = FastAPI() # FastAPI 객체 생성

if __name__ == '__main__' : 
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)

import controller
app.include_router(controller.router)