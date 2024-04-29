import audioread
import librosa
import torch
import torchvision.transforms as T

def preprocess(filename: str, cqt_time_reduction=20) -> None:
    read = audioread.audio_open(filename)
    song, sr = librosa.load(read)
    cqt = np.abs(librosa.cqt(y=song, sr=sr))
    height, length = cqt.shape
    cqt_compressed = cqt[:, :(length//cqt_time_reduction)*cqt_time_reduction].reshape(height, -1, cqt_time_reduction).mean(axis=2)
    transforms = T.Compose([
        lambda x : x.T,
        lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
        lambda x : torch.Tensor(x),
        lambda x : x.permute(1,0).unsqueeze(0),
    ])
    result = transforms(cqt_compressed)
    return result

def search_database(db, names, data):
    distances = torch.pairwise_distance(data, db)
    indices = torch.topk(distances, 10, largest=False).indices.numpy()
    return list(names[indices])