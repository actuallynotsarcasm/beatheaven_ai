from fastapi import APIRouter, Request, Response
from fastapi import File, UploadFile
import traceback
import soundfile
import librosa
import io

import service

router = APIRouter()

@router.get('/')
async def root():
    return 'service up'

@router.post('/find_similar')
async def find_similar(request: Request, response: Response, file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        readable = io.BytesIO(contents)
        song, sr = soundfile.read(readable, channels=1, samplerate=44100, dtype='float32', format='RAW', subtype='PCM_16')
        song, sr = librosa.resample(song, orig_sr=sr, target_sr=22050), 22050
        data = service.preprocess(song, sr)
        output = request.app.state.model(data)[0][0]
        similar_songs = service.search_database(request.app.state.db, request.app.state.song_names, output)
    except Exception:
        response.status_code = 500
        traceback.print_exc()
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    return {"result": similar_songs}