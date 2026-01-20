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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        self.trans_manager = TranslationManager(mode=trans_mode, url=trans_url, ollama_model=ollama_model, target_lang=target_lang) if enable_translate else None
        
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
        self.fragment_queue = queue.Queue() # New queue for raw ASR fragments
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
                        # print(f"[語音結束] ({reason}) 錄製長度: {current_speech_len / 1000:.2f}秒")
                        triggered = False
                        
                        # 檢查總長度是否足夠
                        total_duration_ms = len(frames) * self.CHUNK_DURATION_MS
                        if total_duration_ms > self.min_speech_ms:
                            # 輸出音訊 processing
                            audio_data = b''.join(frames)
                            np_audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                            self.audio_queue.put(np_audio)
                        else:
                            # print("(語音太短，忽略)")
                            pass
                            
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
        """語音辨識執行緒 (Producer)"""
        print("ASR 執行緒啟動")
        
        # 僅用於此線程 Prompt Context 的簡單緩存，真正的句子緩衝移至 text_processing_thread
        # 我們仍保留一些基本的 context 以優化 Whisper 識別準確度，但不處理完整的句子斷句 logic
        self.prev_text_context = "" 

        while self.running:
            try:
                # 從佇列取得音訊 (Blocking)
                audio_data = self.audio_queue.get(timeout=1)
                
                # Identify Speaker
                current_speaker = self.spk_id.identify(audio_data)
                
                # 使用 Whisper 辨識
                # 加入 initial_prompt 提供上下文
                glossary_prompt = self.glossary.get_prompt_context()
                force_tc_prompt = "以下是繁體中文的內容。"
                prev_context = self.prev_text_context[-100:] if self.prev_text_context else ""
                prompt = f"{force_tc_prompt} {glossary_prompt} {prev_context}".strip()
                
                segments, info = self.whisper_model.transcribe(
                    audio_data,
                    language=self.source_lang,
                    beam_size=5,
                    vad_filter=False,
                    initial_prompt=prompt
                )
                
                valid_segments = []
                for segment in segments:
                    if segment.no_speech_prob > 0.95: 
                        continue
                    if segment.avg_logprob < -1.0:
                        continue
                    valid_segments.append(segment.text.strip())

                # 合併所有片段
                current_text = " ".join(valid_segments)
                
                # 應用專有名詞校正
                current_text = self.glossary.correct_text(current_text)
                current_text = self.glossary.clean_text(current_text)
                
                if current_text.strip():
                    # Update local context for next prompt
                    self.prev_text_context += current_text
                    if len(self.prev_text_context) > 200:
                        self.prev_text_context = self.prev_text_context[-200:]

                    # Push fragment to processing queue
                    self.fragment_queue.put({
                        'text': current_text,
                        'speaker': current_speaker,
                        'timestamp': time.time()
                    })

            except queue.Empty:
                continue
            except Exception as e:
                print(f"ASR 錯誤: {e}")

    def text_processing_thread(self):
        """文字處理執行緒 (Consumer) - 負責斷句與緩衝"""
        print("文字處理執行緒啟動")

        self.sentence_endings = {'。', '？', '！', '.', '?', '!'}
        self.text_buffer = ""
        self.buffer_speaker = None
        self.last_buffer_update = time.time()
        
        # 嘗試載入標點復原模型
        self.punct_restorer = PunctuationRestorer()
        
        while self.running:
            try:
                # 嘗試取得新片段
                try:
                    fragment_data = self.fragment_queue.get(timeout=0.5) # Short timeout to allow checking flush logic
                    new_text = fragment_data['text']
                    current_speaker = fragment_data['speaker']
                    
                    # Check for speaker change
                    if self.text_buffer and self.buffer_speaker and current_speaker != self.buffer_speaker:
                         self._flush_buffer(reason="Speaker Switch")
                    
                    if not self.text_buffer:
                        self.buffer_speaker = current_speaker
                        
                    self.text_buffer += new_text
                    self.last_buffer_update = time.time()
                    
                    # 嘗試斷句
                    self._process_buffer()
                    
                except queue.Empty:
                    # Check for timeout flush
                     if self.text_buffer and (time.time() - self.last_buffer_update > 2.0):
                        self._flush_buffer(reason="Timeout")
                        
            except Exception as e:
                print(f"文字處理錯誤: {e}")

    def _process_buffer(self):
        """嘗試從 buffer 中處理出完整句子"""
        # Restore punctuation
        restored_text = self.text_buffer
        if self.punct_restorer.use_punct_model:
            restored_text = self.punct_restorer.restore(self.text_buffer)
            
        # Check if complete
        is_complete = self.punct_restorer.is_complete_sentence(restored_text, self.sentence_endings)
        
        if is_complete:
            final_text = restored_text.strip()
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n✅ [{timestamp}] [{self.buffer_speaker}]: {final_text}")
            
            self.text_queue.put({
                'text': final_text,
                'speaker': self.buffer_speaker
            })
            self.text_buffer = ""
        else:
            # print(f"Buffer (Wait): {restored_text}")
            pass

    def _flush_buffer(self, reason="Force"):
        """強制輸出 Buffer"""
        if not self.text_buffer:
            return
            
        if self.punct_restorer.use_punct_model:
            final_text = self.punct_restorer.restore(self.text_buffer)
        else:
            final_text = self.text_buffer
            
        final_text = final_text.strip()
        if final_text:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n⏰ [{timestamp}] 強制輸出 ({reason}): {final_text}")
            
            self.text_queue.put({
                'text': final_text,
                'speaker': self.buffer_speaker if self.buffer_speaker else "Unknown"
            })
            
        self.text_buffer = ""

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


                if not self.enable_translate:
                    # 如果未開啟翻譯，直接放入隊列但翻譯欄位為空或原樣
                    translation = None
                else:
                    translation = self.trans_manager.translate(text)
                
                # 發送結果到伺服器 (包含翻譯)
                self.notifier.send(text, speaker, translation)
                
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
            threading.Thread(target=self.text_processing_thread, daemon=True), # New Thread
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
    parser.add_argument("--model", type=str, default=os.getenv("WHISPER_MODEL", "medium"), help="Whisper model size (small, medium, large-v2)")
    parser.add_argument("--model_dir", type=str, default=os.getenv("WHISPER_MODEL_DIR"), help="Path to model directory")
    parser.add_argument("--source", type=str, default=os.getenv("SPEECH_SOURCE_LANG", "zh"), help="Source language code")
    parser.add_argument("--target", type=str, default=os.getenv("SPEECH_TARGET_LANG", "en"), help="Target language code")
    
    # Handle boolean for GPU
    use_gpu_env = os.getenv("USE_GPU", "true").lower() == "true"
    parser.add_argument("--gpu", action="store_true", default=use_gpu_env, help="Use GPU if available (default: True)")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Force CPU usage")
    
    # Translation arguments
    # Translation arguments
    enable_translate_env = os.getenv("ENABLE_TRANSLATE", "true").lower() == "true"
    parser.add_argument("--translate", action="store_true", default=enable_translate_env, help="Enable translation (default: True)")
    parser.add_argument("--no-translate", action="store_false", dest="translate", help="Disable translation")
    
    parser.add_argument("--trans_mode", type=str, default=os.getenv("TRANS_MODE", "ollama"), choices=["local", "remote", "ollama"], help="Translation mode: local, remote, or ollama")
    parser.add_argument("--trans_url", type=str, default=os.getenv("TRANS_URL"), help="Remote translation URL (required for remote mode)")
    parser.add_argument("--ollama_model", type=str, default=os.getenv("OLLAMA_MODEL", "hf.co/mradermacher/translategemma-12b-it-GGUF:Q4_K_M"), help="Ollama model name")
    
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
