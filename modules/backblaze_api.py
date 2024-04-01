import asyncio
import aiohttp
import re,datetime
from b2sdk.v2 import InMemoryAccountInfo, B2Api

async def upload_file(application_id, application_key, bucket_name, upload_file_path, db_folder_path, db_file_name: str):
    # B2 APIクライアントの初期化
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", application_id, application_key)

    # バケットの取得
    bucket = b2_api.get_bucket_by_name(bucket_name)

    # 画像ファイルを非同期でアップロード
    async with aiohttp.ClientSession() as session:
        with open(upload_file_path, 'rb') as file:
            file_data = file.read()
            file_path_in_bucket = f"{db_folder_path}/{db_file_name}"
            uploaded_file = await asyncio.to_thread(bucket.upload_bytes, file_data, file_path_in_bucket)

    # ダウンロードURLを取得
    download_url = b2_api.get_download_url_for_file_name(bucket_name, file_path_in_bucket)

    return download_url

async def upload_byte(application_id,application_key,bucket_name,file_data, db_folder_path, db_file_name: str, content_type: str):
    # B2 APIクライアントの初期化
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", application_id,application_key)

    # バケットの取得
    bucket = b2_api.get_bucket_by_name(bucket_name)

    # ファイルを非同期でアップロード
    async with aiohttp.ClientSession() as session:
        file_path_in_bucket = f"{db_folder_path}/{db_file_name}"
        uploaded_file = await asyncio.to_thread(bucket.upload_bytes, file_data, file_path_in_bucket, content_type=content_type)

    # ダウンロードURLを取得
    download_url = b2_api.get_download_url_for_file_name(bucket_name, file_path_in_bucket)
    return download_url

async def download_file(application_id, application_key, bucket_name, file_name_prefix: str, download_auth_token=None):
    # B2 APIクライアントの初期化
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", application_id, application_key)

    # ダウンロードURLの形成
    if download_auth_token:
        download_url = f"{b2_api.get_download_url_for_file_name(bucket_name, file_name_prefix)}?Authorization={download_auth_token}"
    else:
        download_url = f"{b2_api.get_download_url_for_file_name(bucket_name, file_name_prefix)}"

    # ファイルを非同期でダウンロード
    async with aiohttp.ClientSession() as session:
        async with session.get(download_url) as response:
            if response.status == 200:
                file_data = await response.read()
                return file_data
            else:
                raise Exception(f"Failed to download file: HTTP {response.status}")
            
async def list_files_in_folder(application_id,application_key,bucket_name: str, folder_path: str):
    # B2 APIクライアントの初期化
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", application_id,application_key)

    # バケットの取得
    bucket = b2_api.get_bucket_by_name(bucket_name)

    # フォルダ内のファイルリストを非同期で取得
    file_names = []
    async with aiohttp.ClientSession() as session:
        for file_info, folder_name in await asyncio.to_thread(bucket.ls, folder_path, recursive=False):
            if file_info:
                file_names.append(file_info.file_name)
    
    return file_names

async def delete_files_from_folder(application_id,application_key,bucket_name: str, files_to_delete: list):
    """
    指定されたバケット内の特定フォルダーから、リストで指定されたファイルを非同期で削除します。

    Parameters:
    - bucket_name (str): ファイルを削除する対象のバケット名。
    - files_to_delete (list): 削除対象のファイルパスのリスト。

    Returns:
    - list: 各ファイルの削除結果を含むリスト。各要素は、ファイル名と削除の成否（またはエラーメッセージ）のタプルです。

    この関数は、B2 Cloud StorageのAPIを使用して特定のファイルを削除します。削除操作は非同期に行われ、
    操作の結果が各ファイルごとにリスト形式で返されます。これにより、複数のファイルの削除状況を効率的に把握できます。
    """
    # B2 APIクライアントの初期化
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", application_id,application_key)

    # バケットの取得
    bucket = b2_api.get_bucket_by_name(bucket_name)

    # 指定フォルダーからリストで渡されたファイルを非同期で削除
    async with aiohttp.ClientSession() as session:
        delete_results = []
        for file_name in files_to_delete:
            try:
                # ファイルのバージョンIDを取得するためにファイル情報を取得
                file_info = bucket.get_file_info_by_name(file_name)
                # ファイルを削除
                await asyncio.to_thread(bucket.delete_file_version, file_info.id_, file_info.file_name)
                delete_results.append((file_name, "Deleted"))
            except Exception as e:
                delete_results.append((file_name, f"Error: {str(e)}"))

    return delete_results

def create_download_auth_token(application_id,application_key,bucket_name: str, file_name_prefix: str, valid_duration_in_seconds: int):
    """
    特定のファイルまたはファイル群に対する一時的なダウンロードアクセストークンを作成します。このトークンは、指定された期間のみ有効で、指定されたファイル名プレフィックスを持つファイルへのアクセスを許可します。

    Parameters:
    - bucket_name (str): トークンを発行するバケットの名前。
    - file_name_prefix (str): アクセスを許可するファイルの名前プレフィックス。指定したプレフィックスを持つファイルのみがアクセス可能になります。
    - valid_duration_in_seconds (int): トークンの有効期間（秒）。この期間終了後、トークンは無効になります。

    Returns:
    - str: 生成されたダウンロード認証トークン。

    入力例:
    bucket_name = "example-bucket"
    file_name_prefix = "documents/report_"
    valid_duration_in_seconds = 3600  # 1時間

    この例では、"example-bucket"内の"documents/report_"で始まる全てのファイルに対して、1時間有効なダウンロード認証トークンを生成します。
    """
    # B2 APIクライアントの初期化
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", application_id,application_key)
    
    # バケットの取得
    bucket = b2_api.get_bucket_by_name(bucket_name)

    # ダウンロード認証トークンの作成
    download_auth_token = bucket.get_download_authorization(file_name_prefix, valid_duration_in_seconds)
    
    return download_auth_token

async def create_b2_application_key(application_id,application_key,key_name, capabilities, bucket_id=None, file_name_prefix=None, valid_duration_in_seconds=None):
    """
    B2 Cloud Storageで新しいアプリケーションキーを非同期に作成します。

    Parameters:
    - account_id (str): B2アカウントID。
    - application_key (str): B2アカウントのマスターアプリケーションキー。
    - key_name (str): 作成するアプリケーションキーの名前。
    - capabilities (list): キーに割り当てる権限のリスト。例: ['listKeys', 'writeFiles']。
        - listBuckets: バケットのリストを表示する権限。
        - listFiles: バケット内のファイルのリストを表示する権限。
        - readFiles: バケット内のファイルを読み取る権限。
        - shareFiles: ファイル共有（ダウンロード認証トークンの作成）の権限。
        - writeFiles: バケット内にファイルを書き込む権限。
        - deleteFiles: バケット内のファイルを削除する権限。
        - listKeys: アプリケーションキーのリストを表示する権限。
        - writeKeys: 新しいアプリケーションキーを作成する権限。
        - deleteKeys: アプリケーションキーを削除する権限。
        - listAllBucketNames: すべてのバケットの名前をリストする権限。
        - readBucketEncryption: バケットの暗号化設定を読み取る権限。
        - writeBucketEncryption: バケットの暗号化設定を変更する権限。
        - readBucketRetentions: バケットのリテンション（保持期間）ポリシーを読み取る権限。
        - writeBucketRetentions: バケットのリテンションポリシーを変更する権限。
    - bucket_id (str, optional): アクセスを限定したいバケットのID。指定しない場合、キーはアカウントレベルでの権限を持ちます。
    - file_name_prefix (str, optional): アクセスを限定したいファイルの名前のプレフィックス。
    - valid_duration_in_seconds (int, optional): キーの有効期間（秒）。指定しない場合、デフォルトの有効期間が適用されます。

    Returns:
    - dict: 新しいアプリケーションキーの情報を含む辞書。キー作成に失敗した場合は、エラーメッセージが含まれます。

    注意: この関数を使用する前に、B2 Cloud StorageのAPIが`fileNamePrefix`と`validDurationInSeconds`をサポートしていることを確認してください。
    """
    auth_url = 'https://api.backblazeb2.com/b2api/v2/b2_authorize_account'

    async with aiohttp.ClientSession() as session:
        async with session.get(auth_url, auth=aiohttp.BasicAuth(application_id,application_key)) as response:
            data = await response.json()
            api_url = data['apiUrl']  # APIを呼び出すためのURL
            auth_token = data['authorizationToken']  # 認証トークン

            create_key_url = f'{api_url}/b2api/v2/b2_create_key'
            create_key_payload = {
                'accountId': application_id,
                'capabilities': capabilities,
                'keyName': key_name,
                'validDurationInSeconds': valid_duration_in_seconds or 3600,  # デフォルトは1時間
            }

            if bucket_id:
                create_key_payload['bucketId'] = bucket_id
            if file_name_prefix:
                create_key_payload['fileNamePrefix'] = file_name_prefix

            headers = {
                'Authorization': auth_token
            }

            async with session.post(create_key_url, json=create_key_payload, headers=headers) as create_response:
                new_key_data = await create_response.json()
                return new_key_data
            
# ファイル名から日時を抽出し、非同期にdatetimeオブジェクトに変換する関数
async def extract_datetime_from_filename(filename):
    match = re.search(r'\d{14}', filename)
    if match:
        datetime_str = match.group()
        return datetime.datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
    return None

async def find_recent_files(files_list, time_s):
    recent_files = []
    now = datetime.datetime.now()

    # 各ファイルの日時情報を非同期で抽出
    tasks = [asyncio.create_task(extract_datetime_from_filename(filename)) for filename in files_list]
    file_datetimes = await asyncio.gather(*tasks)

    for i, file_datetime in enumerate(file_datetimes):
        if file_datetime:  # 日時情報が存在する場合のみ
            filename = files_list[i]
            if (now - file_datetime).total_seconds() >= time_s:
                recent_files.append(filename)

    return recent_files

async def test():
    import os
    file_name = "temp/20240331172044.pdf"
    bucket_name = 'pdf-private'
    id = os.environ["blackblaze_private_id"]
    key = os.environ["blackblaze_private_key"]
    #auth_token = create_download_auth_token(id,key,'pdf-private',file_name,60)
    list_file_data = await list_files_in_folder(id,key,bucket_name,'temp')
    print(list_file_data)
    list_file_data = await find_recent_files(list_file_data,300)
    print(list_file_data)
    deleteresult=await delete_files_from_folder(id,key,bucket_name,list_file_data)
    print(deleteresult)
    list_file_data = await list_files_in_folder(id,key,bucket_name,'temp')
    print(list_file_data)
    
    #pdf_byte = await download_file(id,key,'pdf-private',file_name,auth_token)

async def test2():
    import os
    file_name = "temp/20240401143247.pdf"
    bucket_name = 'pdf-private'
    id = os.environ["blackblaze_private_id"]
    key = os.environ["blackblaze_private_key"]
    auth_token = create_download_auth_token(id,key,'pdf-private',file_name,60)
    print(f"?Authorization={auth_token}")

if __name__ == "__main__":
    # 非同期タスクを実行
    asyncio.run(test2())