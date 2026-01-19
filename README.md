# SpeechServer

SpeechServer 是一個即時語音辨識、翻譯與轉播系統。即時捕捉語音，進行 VAD (語音活動檢測)、ASR (自動語音辨識)、講者辨識，並透過 WebSocket 將字幕廣播到網頁端與 OBS 直播介面。

## 專案結構

- **server/**: Go 語言編寫的後端伺服器。負責處理 WebSocket 連線、廣播訊息以及提供網頁介面。
- **speech_py/**: Python 編寫的語音處理客戶端。負責音訊採集、VAD 過濾、Whisper 辨識以及講者識別。

## 系統架構

1. **Audio Input**: Python 客戶端透過 PyAudio 擷取麥克風音訊。
2. **Preprocessing**:
   - 使用 **Silero VAD** 過濾靜音片段。
   - 使用 **SpeechBrain** 進行講者識別 (Speaker Identification)。
3. **ASR (Speech-to-Text)**: 使用 **Faster-Whisper** 進行高效能語音轉文字。
4. **Post-processing**:
   - 透過 **GlossaryManager** 進行專有名詞校正。
   - 使用 **DeepMultilingualPunctuation** 進行標點符號還原。
5. **Broadcasting**: 辨識後的文字透過 HTTP POST 發送至 Server，Server 再透過 WebSocket 推播至所有連線的客戶端 (Listener/OBS)。

## 安裝與執行

### 1. Server (後端伺服器)

伺服器使用 Go 語言開發，並支援 Docker 部署。

**前置需求:**
- Docker (推薦) 或 Go 1.16+

**執行步驟:**

進入 `server` 目錄：
```bash
cd server
```

使用 Make 指令執行 (透過 Docker):
```bash
make run
```
預設服務會啟動於 Port `11050`。

若要停止服務：
```bash
make stop
```

---

### 2. Speech Client (語音辨識端)

Python 客戶端，建議在具備 NVIDIA GPU 的環境下執行以獲得最佳效能。

**前置需求:**
- Python 3.8+
- NVIDIA GPU + CUDA Toolkit (建議)

**安裝依賴:**

進入 `speech_py` 目錄並安裝 Python 套件：
```bash
cd speech_py
pip install -r requirements.txt
```
*注意: `pyaudio` 可能需要另外安裝系統層級的依賴 (如 `portaudio`)。*

**執行:**

```bash
python speech.py
```

## 功能特色

- **專業級 VAD**: 整合 Silero VAD，精準過濾非語音噪音 (如咳嗽、清喉嚨)。
- **高效 ASR**: 採用 Faster-Whisper 模型，支援 GPU 加速。
- **講者識別**: 自動區分不同發話者 (Speaker 0, Speaker 1...)。
- **即時字幕**:
  - **Web Client**: 聽眾可透過瀏覽器即時觀看字幕與翻譯。
  - **OBS Overlay**: 提供專用的 HTML 頁面供 OBS 下載使用，支援透明背景字幕。
- **術語表 (Glossary)**: 支援自定義術語替換，提升專有名詞辨識率。
- **自動標點**: AI 模型自動回復語句標點，提升閱讀體驗。

## 設定

- **Python Client**: 在 `speech.py` 中可調整 `WhisperModel` 大小、語言設定 (`source_lang`, `target_lang`) 以及伺服器 URL。
- **Server**: 可透過 `server/envfile` 或 `makefile` 修改 Port 與資料庫設定。

## 聲音通道
你需要的流程是： 瀏覽器 -> 喇叭輸出 (Output Sink) -> 監聽端 (Monitor Source) -> Python
