import os
import soxr
import json
import torch
import soundfile as sf
import numpy as np

from tqdm import tqdm
from scipy.special import softmax
from python_speech_features import logfbank


mdl_bad_kwargs = {
    "channels": 16, 
    "block": "BasicBlock", 
    "num_blocks": [2,2,2,2], 
    "embd_dim": 1024, 
    "drop": 0.3, 
    "n_class": 821
}

logfbank_kwargs = {
    "winlen": 0.025, 
    "winstep": 0.01, 
    "nfilt": 80, 
    "nfft": 2048, 
    "lowfreq": 50, 
    "highfreq": None, 
    "preemph": 0.97    
}

with open('label2int.json','r') as f:
    label2int = json.load(f)
    int2label = {v:k for k,v in label2int.items()}
model_dir = os.path.join(os.path.dirname(__file__), '821b_jit.pt')


def extract_feat(wav_path, samplerate=16000, cmn=True):
    kwargs = {
        "winlen": 0.025,
        "winstep": 0.01,
        "nfilt": 80,
        "nfft": 2048,
        "lowfreq": 50,
        "highfreq": 8000,
        "preemph": 0.97
    }
    y, sr = sf.read(wav_path)
    if sr!=samplerate:
        y = soxr.resample(y, sr, samplerate)
        sr = samplerate
    logfbankFeat = logfbank(y, sr, **kwargs)
    if cmn:
        logfbankFeat -= logfbankFeat.mean(axis=0, keepdims=True)
    return logfbankFeat.astype('float32')
    

class SVExtractor():
    def __init__(self, mdl_kwargs, model_path, device):
        self.model = self.load_model(mdl_kwargs, model_path, device)
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)

    def load_model(self, mdl_kwargs, model_path, device):
        model = torch.jit.load(model_path, map_location=device)
        return model

    def __call__(self, frame_feats):
        feat = torch.from_numpy(frame_feats).unsqueeze(0)
        feat = feat.float().to(self.device)
        with torch.no_grad():
            embd = self.model(feat)
        embd = embd.squeeze(0).cpu().numpy()
        return embd
    

def from_wav(wav, device="cpu"):
    """
    params:
        wav:             string; wave file for bird activity detection
        top_k:           int; top k possible predictions [default=1]
        device:          string; device for calculation [default="cpu"] [options: ['cpu','cuda:0']]
        
    return:
        species:         prediction of input audio
        logits:          confidence for the prediction
    """
    wav_feats = extract_feat(wav)
    detector = SVExtractor(mdl_bad_kwargs, model_dir, device=device)
    logits = softmax(detector(wav_feats)).tolist()
    top_pred = np.argmax(logits)
    confidence = logits[top_pred]
    return int2label[top_pred], confidence
    
def from_wavs(wavs, device="cpu"):
    """
    params:
        wav:             wave file for bird activity detection
        top_k:           int; top k possible predictions [default=1]
        device:          device for calculation [default="cpu"] [options: ['cpu','cuda:0']]
    
    return:
        results:         List(Tuples(utt_name, species, logits))
    """
    detector = SVExtractor(mdl_bad_kwargs, model_dir, device=device)
    results = []
    for wav_ in tqdm(wavs, desc="Detecting noises"):
        utt_ = wav_.split('/')[-1].split('.')[0]
        wav_feats = extract_feat(wav_)
        wav_logits = softmax(detector(wav_feats)).tolist()
        wav_pred = np.argmax(wav_logits)
        wav_confidence = wav_logits[wav_pred]
        results.append((utt_, int2label[wav_pred], wav_confidence))
    return results