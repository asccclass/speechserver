
import torch
import numpy as np

class SileroVAD:
    def __init__(self):
        # Load Silero VAD model
        print("載入 Silero VAD 模型...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        self.model.eval()
        print("Silero VAD 模型載入完成")

    def is_speech(self, audio_chunk_int16, sampling_rate=16000):
        # Convert int16 bytes/array to float32 tensor normalized to [-1, 1]
        if isinstance(audio_chunk_int16, bytes):
            audio_int16 = np.frombuffer(audio_chunk_int16, dtype=np.int16)
        else:
            audio_int16 = audio_chunk_int16
            
        # Normalization: int16 -> float32 [-1, 1]
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # Determine strictness (optional, model output is probability)
        # We handle tensor conversion here
        tensor = torch.from_numpy(audio_float32)
        
        # Add batch dimension if needed, Silero expects (batch, time) or (time)
        # but for single chunk (512), 1D is fine.
        
        with torch.no_grad():
            speech_prob = self.model(tensor, sampling_rate).item()
            
        return speech_prob
