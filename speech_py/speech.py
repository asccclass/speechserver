"""
即時演講翻譯系統
架構: ASR → Translation → Display/TTS
使用 faster-whisper + OpenAI/Claude API + 字幕顯示
"""

import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*Found GPU.*cuda capability.*")
warnings.filterwarnings("ignore", message=".*grouped_entities.*")

import pyaudio
import numpy as np
import threading
import queue
from faster_whisper import WhisperModel
from datetime import datetime
import time
import collections
from contextlib import contextmanager
import os
import sys
from vad import SileroVAD
from glossarymanager import GlossaryManager
from speakerid import SpeakerIdentifier
from punmarks import PunctuationRestorer
from translate import TranslationManager
from notifyserver import ServerNotifier
import argparse

@contextmanager
def ignore_stderr():
    """Suppress stderr output (useful for hiding ALSA/PyAudio warnings)"""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
    except Exception:
        # If stderr redirection fails (e.g. on some Windows environments), just yield
        yield

class RealtimeSpeechTranslator:
    def __init__(self, 
                 source_lang="zh",
                 target_lang="en",
                 whisper_model="medium",
                 use_gpu=True,
                 model_dir=None,
                 enable_translate=False,
                 enable_translate=False,
                 trans_mode='local',
                 trans_url=None,
                 ollama_model=None):
        """
        初始化即時翻譯系統
        
        Args:
            source_lang: 來源語言 (zh, en, ja, ko, etc.)
            target_lang: 目標語言
            whisper_model: Whisper 模型大小 (tiny, base, small, medium, large)
            use_gpu: 是否使用 GPU
            model_dir: Whisper 模型下載/讀取路徑 (Optional)
            enable_translate: 是否開啟翻譯功能
            trans_mode: 翻譯模式 ('local', 'remote', 'ollama')
            trans_url: 遠端翻譯 API URL
            ollama_model: Ollama 模型名稱
        """
        # 音訊參數
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        # VAD 參數
        self.CHUNK_DURATION_MS = 32  # Silero VAD requires 32ms (512 samples) or 64ms (1024 samples) at 16k
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION_MS / 1000)  # 512 samples
        
        # Initialize Professional VAD
        self.vad_model = SileroVAD()
        
        # Initialize Speaker Identifier
        self.spk_id = SpeakerIdentifier()
        
        # Initialize Glossary Manager
        self.glossary = GlossaryManager()

        # Initialize Server Notifier
        self.notifier = ServerNotifier()
        
        # 語言設定
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # 翻譯設定
        self.enable_translate = enable_translate
        self.trans_manager = TranslationManager(mode=trans_mode, url=trans_url, ollama_model=ollama_model) if enable_translate else None
        
        # 載入 Whisper 模型
        print(f"載入 Whisper 模型: {whisper_model}")
        device = "cuda" if use_gpu else "cpu"
        compute_type = "float16" if use_gpu else "int8"
        
        try:
            self.whisper_model = WhisperModel(
                whisper_model, 
                device=device, 
                compute_type=compute_type,
                download_root=model_dir
            )
        except Exception as e:
            print(f"Whisper 模型載入失敗，嘗試使用 CPU int8: {e}")
            self.whisper_model = WhisperModel(
                whisper_model, 
                device="cpu", 
                compute_type="int8",
                download_root=model_dir
            )
        
        # 佇列和狀態
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.translation_queue = queue.Queue()
        self.running = False
        
        # VAD 閾值設定 (動態調整或固定)
        self.vad_threshold = 0.6      # Silero Probability threshold (Increased for stricter speech filtering)
        self.speech_pad_ms = 800      # 語音前後的緩衝時間 (ms) - Increased for buffer
        self.min_speech_ms = 500      # 最短語音長度 (ms)
        self.max_silence_ms = 1200    # 語音中間允許的最長靜音 (ms) - Increased to allow pauses
        
        # Dynamic Buffering Strategy
        self.dynamic_silence_ms = 600   # Aggressive silence threshold for long segments (ms)
        self.long_speech_ms = 15000     # Threshold to trigger aggressive completion (15s)
        self.force_speech_ms = 40000    # Hard limit to force cut (40s)
        
        # 翻譯緩存 (避免重複翻譯)
        self.translation_cache = {}
        
    def _is_speech(self, audio_chunk):
        """計算 VAD Probability"""
        # audio_chunk 是 bytes
        # Using Silero VAD
        speech_prob = self.vad_model.is_speech(audio_chunk, self.RATE)
        return speech_prob > self.vad_threshold, speech_prob

    def audio_capture_thread(self):
        """
        基於 VAD 的智慧錄音循環
        流程: 靜音 -> 偵測到聲音 -> 錄製中 -> 靜音超時 -> 輸出片段
        """
        with ignore_stderr():
            p = pyaudio.PyAudio()
        stream = p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE
        )
        
        print(f"開始監聽... (閾值: {self.vad_threshold})")
        print("請說話...")
        
        # 狀態變數
        triggered = False
        frames = []
        silence_duration_ms = 0
        ring_buffer = collections.deque(maxlen=int(self.speech_pad_ms / self.CHUNK_DURATION_MS))
        
        while self.running:
            try:
                data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                is_speech, rms = self._is_speech(data)
                
                # 簡單的動態閾值調整 (可選，這裡只印出 debug)
                # if not triggered and rms > 10: print(f"Current RMS: {rms:.1f}", end='\r')

                if not triggered:
                    ring_buffer.append(data)
                    if is_speech:
                        print(f"\n[偵測到語音] RMS: {rms:.1f} 開始錄製...")
                        triggered = True
                        frames.extend(ring_buffer) # 加入緩衝的前段聲音
                        frames.append(data)
                        silence_duration_ms = 0
                else:
                    frames.append(data)
                    if is_speech:
                        silence_duration_ms = 0
                    else:
                        silence_duration_ms += self.CHUNK_DURATION_MS
                        
                    # Calculate current length
                    current_speech_len = len(frames) * self.CHUNK_DURATION_MS
                    
                    # Determine effective silence threshold based on length
                    effective_silence_threshold = self.max_silence_ms
                    if current_speech_len > self.long_speech_ms:
                        effective_silence_threshold = self.dynamic_silence_ms
                        
                    # Check for cut conditions:
                    # 1. Silence exceeded threshold
                    # 2. Total length exceeded hard limit
                    should_cut = (silence_duration_ms > effective_silence_threshold)
                    force_cut = (current_speech_len > self.force_speech_ms)
                    
                    if should_cut or force_cut:
                        reason = "Max silence" if should_cut else "Force cut"
                        print(f"[語音結束] ({reason}) 錄製長度: {current_speech_len / 1000:.2f}秒")
                        triggered = False
                        
                        # 檢查總長度是否足夠
                        total_duration_ms = len(frames) * self.CHUNK_DURATION_MS
                        if total_duration_ms > self.min_speech_ms:
                            # 輸出音訊 processing
                            audio_data = b''.join(frames)
                            np_audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                            self.audio_queue.put(np_audio)
                        else:
                            print("(語音太短，忽略)")
                            
                        # 重置
                        frames = []
                        ring_buffer.clear()
            
            except Exception as e:
                print(f"錄音錯誤: {e}")
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

    def asr_thread(self):
        """語音辨識執行緒"""
        print("ASR 執行緒啟動")
        

        self.sentence_endings = {'。', '？', '！', '.', '?', '!'}
        self.text_buffer = ""
        self.prev_text = ""  # 上一句確認的文字 (用作 Prompt context)
        self.last_buffer_update = time.time()
        self.buffer_speaker = None
        
        # 嘗試載入標點復原模型
        self.punct_restorer = PunctuationRestorer()

        while self.running:
            try:
                # 從佇列取得音訊 (Blocking)
                audio_data = self.audio_queue.get(timeout=1)
                
                print(f"正在辨識... (長度: {len(audio_data)/16000:.1f}s)")

                # Identify Speaker
                current_speaker = self.spk_id.identify(audio_data)
                self.last_known_speaker = current_speaker
                print(f"[{current_speaker}] 正在發言...")
                
                # Check for speaker change
                if self.text_buffer and self.buffer_speaker and current_speaker != self.buffer_speaker:
                    print(f"\n🔁 [Speaker Change] {self.buffer_speaker} -> {current_speaker}. Flushing buffer.")
                    
                    if self.punct_restorer.use_punct_model:
                        final_text = self.punct_restorer.restore(self.text_buffer)
                    else:
                        final_text = self.text_buffer
                        
                    final_text = final_text.strip()
                    if final_text:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"✅ [{timestamp}] (Speaker Switch): {final_text}")
                        self.text_queue.put({
                            'text': final_text,
                            'speaker': self.buffer_speaker
                        })
                        self.prev_text = final_text
                    
                    self.text_buffer = ""
                
                if not self.text_buffer:
                    self.buffer_speaker = current_speaker
                
                # 使用 Whisper 辨識
                # 加入 initial_prompt 提供上下文，減少幻覺並維持連貫性
                # 結合前文與專有名詞
                glossary_prompt = self.glossary.get_prompt_context()
                prev_context = self.prev_text[-100:] if self.prev_text else "請將語音辨識為繁體中文。"
                prompt = f"{glossary_prompt} {prev_context}".strip()
                
                segments, info = self.whisper_model.transcribe(
                    audio_data,
                    language=self.source_lang,
                    beam_size=5,
                    vad_filter=False,
                    initial_prompt=prompt
                )
                
                # Filter segments based on confidence to remove noise (coughing, throat clearing)
                valid_segments = []
                for segment in segments:
                    # no_speech_prob: Probability that the segment contains no speech
                    # avg_logprob: Average log probability (confidence) of the text
                    if segment.no_speech_prob > 0.95: 
                        print(f"🙈 過濾雜音 (No Speech Prob: {segment.no_speech_prob:.2f}): {segment.text}")
                        continue
                    if segment.avg_logprob < -1.0: # Configurable threshold
                        print(f"🙈 過濾低信度 (LogProb: {segment.avg_logprob:.2f}): {segment.text}")
                        continue
                    valid_segments.append(segment.text.strip())

                # 合併所有片段
                current_text = " ".join(valid_segments)
                
                # 應用專有名詞校正
                current_text = self.glossary.correct_text(current_text)
                current_text = self.glossary.clean_text(current_text)
                
                if current_text.strip():
                    print(f"片段識別: {current_text}")
                    self.text_buffer += current_text
                    self.last_buffer_update = time.time()
                    
                    # 處理標點與斷句
                    restored_text = self.text_buffer
                    if self.punct_restorer.use_punct_model:
                        restored_text = self.punct_restorer.restore(self.text_buffer)
                    
                    is_complete = self.punct_restorer.is_complete_sentence(restored_text, self.sentence_endings)
                    
                    if is_complete:
                        final_text = restored_text.strip()
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"\n✅ [{timestamp}] : {final_text}")
                        
                        self.text_queue.put({
                            'text': final_text,
                            'speaker': current_speaker
                        })
                        self.prev_text = final_text
                        self.text_buffer = ""
                    else:
                        print(f"等待完整句子 (Restored: {restored_text})...")

                else:
                    print("❌ 無法識別出文字")
                    
            except queue.Empty:
                # 超時機制：如果太久沒有新聲音(2秒)，且緩衝區有字，強制輸出
                if self.text_buffer and (time.time() - self.last_buffer_update > 2.0):
                    
                    if self.punct_restorer.use_punct_model:
                        final_text = self.punct_restorer.restore(self.text_buffer)
                    else:
                        final_text = self.text_buffer
                        
                    final_text = final_text.strip()
                    if final_text:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"\n⏰ [{timestamp}] 超時強制輸出: {final_text}")
                        # 超時強制輸出時，speaker 可能需要用最近一次的，或 Unknown
                        # 這裡暫時無法取得完美的 speaker context，若 audio capture thread 有保留 speaker info 會更好
                        # 但既然是 buffer 殘留，通常是同一個人
                        # 簡化起見，這裡不重新 identify (因為沒有 audio data了)，
                        # 我們可以存一個 self.last_speaker
                        
                        last_speaker = getattr(self, 'last_known_speaker', "Speaker ?")
                        self.text_queue.put({
                            'text': final_text,
                            'speaker': last_speaker
                        })
                        self.prev_text = final_text
                        self.text_buffer = ""
                continue
            except Exception as e:
                print(f"ASR 錯誤: {e}")

    def translation_thread(self):
        """翻譯執行緒"""
        print("翻譯執行緒啟動")
        
        while self.running:
            try:
                item = self.text_queue.get(timeout=1)
                if isinstance(item, dict):
                    text = item['text']
                    speaker = item.get('speaker', "Speaker ?")
                else:
                    text = item
                    speaker = "Speaker ?"


                # 1. 恢復原本的發送動作 (Broadcast)
                self.notifier.send(text, speaker)

                # 2. 翻譯處理 (Translation)
                if not self.enable_translate:
                    # 如果未開啟翻譯，直接放入隊列但翻譯欄位為空或原樣，
                    # 但根據需求: "若是沒有開啟翻譯功能，則輸出 譯文 的部分，不用顯示"
                    # 我們這裡可以設為 None
                    translation = None
                else:
                    translation = self.trans_manager.translate(text)
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                self.translation_queue.put({
                    'original': text,
                    'translation': translation,
                    'speaker': speaker,
                    'timestamp': timestamp
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"翻譯錯誤: {e}")
    
    def display_thread(self):
        """顯示執行緒"""
        print("顯示執行緒啟動")
        
        while self.running:
            try:
                result = self.translation_queue.get(timeout=1)
                
                print("\n" + "="*60)
                print(f"時間: {result['timestamp']}")
                print(f"講者: {result['speaker']}")
                print(f"原文: {result['original']}")
                if result['translation']:
                    print(f"譯文: {result['translation']}")
                print("="*60 + "\n")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"顯示錯誤: {e}")
    
    def start(self):
        """啟動系統"""
        print("\n===== 專業版即時演講翻譯系統 (VAD Enabled) =====")
        print(f"來源語言: {self.source_lang}")
        print(f"目標語言: {self.target_lang}")
        print(f"VAD 閾值: {self.vad_threshold}")
        print("按 Ctrl+C 停止\n")
        
        self.running = True
        
        threads = [
            threading.Thread(target=self.audio_capture_thread, daemon=True),
            threading.Thread(target=self.asr_thread, daemon=True),
            threading.Thread(target=self.translation_thread, daemon=True),
            threading.Thread(target=self.display_thread, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n正在停止系統...")
            self.running = False
            for thread in threads:
                thread.join(timeout=2)
            print("系統已停止")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Realtime Speech Translator")
    
    # Mode arguments
    parser.add_argument("--model", type=str, default="medium", help="Whisper model size (small, medium, large-v2)")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to model directory")
    parser.add_argument("--source", type=str, default="zh", help="Source language code")
    parser.add_argument("--target", type=str, default="en", help="Target language code")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU if available (default: True)")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Force CPU usage")
    
    # Translation arguments
    parser.add_argument("--translate", action="store_true", help="Enable translation")
    parser.add_argument("--trans_mode", type=str, default="local", choices=["local", "remote", "ollama"], help="Translation mode: local, remote, or ollama")
    parser.add_argument("--trans_url", type=str, default=None, help="Remote translation URL (required for remote mode)")
    parser.add_argument("--ollama_model", type=str, default="hf.co/mradermacher/translategemma-12b-it-GGUF:Q4_K_M", help="Ollama model name")
    
    args = parser.parse_args()

    translator = RealtimeSpeechTranslator(
        source_lang=args.source,
        target_lang=args.target,
        whisper_model=args.model, 
        use_gpu=args.gpu,
        model_dir=args.model_dir,
        enable_translate=args.translate,
        trans_mode=args.trans_mode,
        trans_url=args.trans_url,
        ollama_model=args.ollama_model
    )
    translator.start()
