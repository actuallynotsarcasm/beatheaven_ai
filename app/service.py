import librosa
import torch
import numpy as np
import torchvision.transforms as T
import os


def _cut_data(data, out_length=394, pad_to=None):
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            offset = np.random.randint(max_offset)
            data = data[offset:(out_length+offset),:]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0,offset),(0,0)), "constant")
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0,offset),(0,0)), "constant")
        
    if pad_to:
        data = np.pad(data, ((0,pad_to-data.shape[0]),(0,0)), "constant")
        
    return data

def preprocess(song, sr, cqt_time_reduction=20) -> None:
    cqt = np.abs(librosa.cqt(y=song, sr=sr))
    height, length = cqt.shape
    cqt_compressed = cqt[:, :(length//cqt_time_reduction)*cqt_time_reduction].reshape(height, -1, cqt_time_reduction).mean(axis=2)
    transforms = T.Compose([
        lambda x : x.T,
        lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
        _cut_data,
        lambda x : torch.Tensor(x),
        lambda x : x.permute(1,0).unsqueeze(0),
    ])
    result = transforms(cqt_compressed)
    return result

def search_database(db, song_names, data):
    distances = torch.pairwise_distance(data[None, :], db)
    indices = torch.topk(distances, 10, largest=False).indices.numpy()
    return list(song_names[indices])