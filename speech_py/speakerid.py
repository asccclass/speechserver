
class SpeakerIdentifier:
    def __init__(self):
        print("載入 Speaker Identification 模型 (SpeechBrain)...")
        try:
            import torchaudio
            # Patch torchaudio for compatibility with speechbrain
            if not hasattr(torchaudio, 'list_audio_backends'):
                torchaudio.list_audio_backends = lambda: ['soundfile']
            
            from speechbrain.inference.classifiers import EncoderClassifier
            import torch
            import torch.nn.functional as F
            self.F = F
            self.torch = torch
            # 使用 ECAPA-TDNN 模型
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            print("Speaker Identification 模型載入完成")
            self.enabled = True
        except Exception as e:
            print(f"⚠️ Speaker Identification 模型載入失敗: {e}")
            print("請確認已安裝 speechbrain: pip install speechbrain")
            self.enabled = False

        self.speakers = {} # {id: {'embedding': tensor, 'count': int}}
        self.next_id = 0
        self.similarity_threshold = 0.50 

    def identify(self, audio_data):
        if not self.enabled:
            return "Speaker ?"
            
        try:
            # audio_data: np.float32 array
            tensor = self.torch.from_numpy(audio_data).unsqueeze(0) # (1, T)
            if self.torch.cuda.is_available():
                tensor = tensor.cuda()
                
            # 取得 Embedding
            embeddings = self.classifier.encode_batch(tensor)
            emb = embeddings.squeeze() 

            best_score = -1.0
            best_speaker = None

            # 比對現有講者
            for spk_id, data in self.speakers.items():
                known_emb = data['embedding']
                score = self.F.cosine_similarity(emb, known_emb, dim=0).item()
                if score > best_score:
                    best_score = score
                    best_speaker = spk_id
            
            if best_score > self.similarity_threshold:
                # Update Embedding
                count = self.speakers[best_speaker]['count']
                old_emb = self.speakers[best_speaker]['embedding']
                new_emb = (old_emb * count + emb) / (count + 1)
                self.speakers[best_speaker]['embedding'] = new_emb
                self.speakers[best_speaker]['count'] = count + 1
                return f"Speaker {best_speaker}"
            else:
                new_id = self.next_id
                self.next_id += 1
                self.speakers[new_id] = {'embedding': emb, 'count': 1}
                return f"Speaker {new_id}"

        except Exception as e:
            print(f"辨識講者錯誤: {e}")
            return "Speaker Err"
