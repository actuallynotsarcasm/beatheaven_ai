from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch
import pandas
import numpy
import uvicorn
import app

from router import router
from model import CQTNet

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = torch.load('data/model.pth', map_location=torch.device('cpu')).module.cpu()
    app.state.db = torch.load('data/db.pth', map_location=torch.device('cpu'))
    app.state.song_ids = pandas.read_csv('data/song_ids.csv')['id']
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)