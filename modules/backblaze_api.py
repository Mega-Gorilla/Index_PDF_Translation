import asyncio
import aiohttp
import os
from b2sdk.v2 import InMemoryAccountInfo, B2Api

# Backblaze B2の認証情報blackblaze_key
application_key_id = os.environ["blackblaze_id"]
application_key = os.environ["blackblaze_key"]
bucket_name = 'pdf-public'

async def upload_file(upload_file_path, db_folder_path, db_file_name: str):
    # B2 APIクライアントの初期化
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", application_key_id, application_key)

    # バケットの取得
    bucket = b2_api.get_bucket_by_name(bucket_name)

    # 画像ファイルを非同期でアップロード
    async with aiohttp.ClientSession() as session:
        with open(upload_file_path, 'rb') as file:
            file_data = file.read()
            file_path_in_bucket = f"{db_folder_path}/{db_file_name}"
            uploaded_file = await asyncio.to_thread(bucket.upload_bytes, file_data, file_path_in_bucket)

    # ダウンロードURLを取得
    download_url = b2_api.get_download_url_for_fileid(uploaded_file.id_)

    return download_url

async def upload_byte(file_data, db_folder_path, db_file_name: str, content_type: str):
    # B2 APIクライアントの初期化
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", application_key_id, application_key)

    # バケットの取得
    bucket = b2_api.get_bucket_by_name(bucket_name)

    # ファイルを非同期でアップロード
    async with aiohttp.ClientSession() as session:
        file_path_in_bucket = f"{db_folder_path}/{db_file_name}"
        uploaded_file = await asyncio.to_thread(bucket.upload_bytes, file_data, file_path_in_bucket, content_type=content_type)

    # ダウンロードURLを取得
    download_url = b2_api.get_download_url_for_fileid(uploaded_file.id_)
    return download_url

if __name__ == "__main__":
    # 非同期タスクを実行
    asyncio.run(upload_file())