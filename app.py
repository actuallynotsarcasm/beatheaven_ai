from fastapi import FastAPI
from contextlib import 
import torch
import uvicorn

from router import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = torch.load('model.pth')
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)