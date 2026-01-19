import requests
from datetime import datetime

class ServerNotifier:
    def __init__(self, url="https://speech.justdrink.com.tw/speaker"):
        self.url = url

    def send(self, text, speaker="Speaker ?", translation=None):
        """
        將辨識結果發送到演講伺服器 (Broadcast)
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            payload = {
                "text": text,
                "speaker": speaker,
                "timestamp": timestamp
            }
            if translation:
                payload["translation"] = translation
            
            # 使用 Session 或直接 post
            response = requests.post(self.url, json=payload, timeout=3)
            
            if response.status_code == 200:
                print(f"✅ 已發送至伺服器: {text[:20]}...")
            else:
                print(f"⚠️ 發送失敗 [{response.status_code}]: {response.text}")
                
        except Exception as e:
            print(f"⚠️ 連線錯誤: {e}")
