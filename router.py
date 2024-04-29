from fastapi import APIRouter, Request, Response, status
from fastapi.responses import RedirectResponse
from fastapi import File, UploadFile
import aiohttp

import service

router = APIRouter()

@router.post('/find_similar')
async def find_similar(file: UploadFile = File(...), request: Request):
    try:
        with open(file.file, rb) as f:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}