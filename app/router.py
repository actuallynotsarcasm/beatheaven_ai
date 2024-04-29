from fastapi import APIRouter, Request, Response
from fastapi import File, UploadFile
import subprocess

import service

router = APIRouter()

@router.get('/')
async def root():
    return 'service up'

@router.post('/find_similar')
async def find_similar(request: Request, response: Response, file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        files_path = 'song_buffer/'
        with open(files_path + file.filename, 'wb') as f:
            f.write(contents)
        subprocess.call(['ffmpeg', '-i', files_path + file.filename, files_path + 'converted.wav', '-y'], 
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        data = service.preprocess(files_path + 'converted.wav')
        service.clear_song_buffer()
        output = request.app.state.model(data)[0][0]
        similar_songs = service.search_database(request.app.state.db, request.app.state.song_names, output)
    except Exception as e:
        raise e
        response.status_code = 500
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    return similar_songs