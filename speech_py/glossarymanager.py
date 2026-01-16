import json
import os
import re

class GlossaryManager:
    def __init__(self, glossary_path="glossary.json"):
        self.glossary_path = glossary_path
        self.terms = []
        self.corrections = {}
        self.load_glossary()
        
    def load_glossary(self):
        if os.path.exists(self.glossary_path):
            try:
                with open(self.glossary_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.terms = data.get("terms", [])
                    self.corrections = data.get("corrections", {})
                print(f"專有名詞庫載入完成: {len(self.terms)} 個詞彙, {len(self.corrections)} 個校正規則")
            except Exception as e:
                print(f"專有名詞庫載入失敗: {e}")
        else:
            print("未找到專有名詞庫 (glossary.json)，將使用預設設定")

    def get_prompt_context(self):
        """產生 Prompt Context"""
        if not self.terms:
            return ""
        return "Keywords: " + ", ".join(self.terms) + "."

    def correct_text(self, text):
        """執行文字校正"""
        if not text: return text
        
        # 簡單的字串替換 (更高級的可以用 Regex 或 Fuzzy matching)
        for wrong, correct in self.corrections.items():
            if wrong in text:
                text = text.replace(wrong, correct)
        return text

    def clean_text(self, text):
        """
        簡易過濾重複字詞
        e.g. "我我我想要" -> "我想要"
        """
        if not text: return ""
        
        # 去除連續重複的中文單字 (e.g. 我我 -> 我)
        text = re.sub(r'([\u4e00-\u9fa5])\1+', r'\1', text)
        return text
