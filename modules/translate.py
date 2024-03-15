import aiohttp
import os

async def translate_text(text: str, target_lang: str) -> str:
    """
    DeepL APIを使用して、入力されたテキストを指定の言語に翻訳する非同期関数。

    Args:
        text (str): 翻訳するテキスト。
        target_lang (str): 翻訳先の言語コード（例: "EN", "JA", "FR"など）。

    Returns:
        str: 翻訳されたテキスト。

    Raises:
        Exception: APIリクエストが失敗した場合。
    """
    api_key = os.environ["DEEPL_API_KEY"]  # 環境変数からDeepL APIキーを取得
    api_url = "https://api-free.deepl.com/v2/translate"

    params = {
        "auth_key": api_key,
        "text": text,
        "target_lang": target_lang
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, data=params) as response:
            if response.status == 200:
                result = await response.json()
                return {'ok':True,'data':result["translations"][0]["text"]}
            else:
                return {'ok':False,'message':f"DeepL API request failed with status code {response.status}"}
            

if __name__ == "__main__":
    import asyncio
    async def main():
        text = "Hello, how are you?"
        translated = await translate_text(text, "JA")
        print(translated)

    asyncio.run(main())