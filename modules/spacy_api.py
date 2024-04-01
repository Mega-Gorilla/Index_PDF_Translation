import spacy
from config import SPACY_CONFIG

# サポートされている言語とそのモデルのマッピング
supported_languages = SPACY_CONFIG['supported_languages']

# spaCyモデルをロードし、キャッシュしておく辞書
loaded_models = {}

def load_model(lang_code):
    """指定された言語コードのモデルをロードする"""
    if lang_code in loaded_models:
        return loaded_models[lang_code]
    if lang_code in supported_languages:
        model_name = supported_languages[lang_code]
        try:
            nlp = spacy.load(model_name)
            loaded_models[lang_code] = nlp
            return nlp
        except OSError as e:
            print(f"Model for '{lang_code}' could not be loaded: {e}")
            return None
    else:
        print(f"No model available for language code: '{lang_code}'")
        return None

def tokenize_text(lang_code, text):
    """指定された言語のテキストをトークン化し、トークンのリストを返す"""
    nlp = load_model(lang_code)
    if nlp:
        doc = nlp(text)
        tokens = [token.text for token in doc if token.is_alpha] #トークンがアルファベットで構成されている可動はも判定
        return tokens
    else:
        return []
    

if __name__ == "__main__":
    # 使用例
    text_en = "This is an English sentence."
    tokens_en = tokenize_text('en', text_en)
    print(f"English text tokens: {tokens_en}")

    text_ja = "これは日本語の文です。"
    tokens_ja = tokenize_text('ja', text_ja)
    print(f"Japanese text tokens: {tokens_ja}")