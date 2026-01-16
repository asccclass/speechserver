
class PunctuationRestorer:
    def __init__(self, model_name="oliverguhr/fullstop-punctuation-multilingual-sonar-base"):
        self.use_punct_model = False
        print("載入標點復原模型 (BERT)...")
        try:
            from deepmultilingualpunctuation import PunctuationModel
            self.punct_model = PunctuationModel(model=model_name)
            print("標點模型載入完成")
            self.use_punct_model = True
        except ImportError:
            print("⚠️ 警告: 未安裝 deepmultilingualpunctuation，將無法自動加標點。")
            print("請執行: pip install deepmultilingualpunctuation")
        except Exception as e:
            print(f"⚠️ 標點模型載入失敗: {e}")

    def restore(self, text):
        if not self.use_punct_model or not text:
            return text
        try:
            return self.punct_model.restore_punctuation(text)
        except Exception as e:
            print(f"標點復原錯誤: {e}")
            return text

    def is_complete_sentence(self, text, sentence_endings=None):
        if sentence_endings is None:
            sentence_endings = {'。', '？', '！', '.', '?', '!'}
        
        text = text.strip()
        if not text:
            return False
            
        return (any(text.endswith(p) for p in sentence_endings) or len(text) > 100)
