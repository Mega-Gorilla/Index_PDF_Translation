import aiohttp
import asyncio
import xml.etree.ElementTree as ET
import time

# レート制限の設定
RATE_LIMIT = 3  # 3秒に1回のリクエスト
# 最後にリクエストを送信した時刻を保持する変数
last_request_time = 0

async def get_arxiv_info_async(arxiv_id):
    
    global last_request_time

    # 現在の時刻を取得
    current_time = time.time()
    
    # 最後のリクエストからの経過時間を計算
    elapsed_time = current_time - last_request_time
    
    # 最後のリクエストから3秒以内の場合、残りの時間だけ待機
    if elapsed_time < RATE_LIMIT:
        await asyncio.sleep(RATE_LIMIT - elapsed_time)

    # 最後のリクエスト時刻を更新
    last_request_time = time.time()

    # OAI-PMH endpoint
    base_url = 'http://export.arxiv.org/oai2'

    # Set parameters for the OAI-PMH request
    params = {
        'verb': 'GetRecord',
        'identifier': f'oai:arXiv.org:{arxiv_id}',
        'metadataPrefix': 'arXiv',
    }

    # Send the OAI-PMH request using aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            # Read the response text
            response_text = await response.text()

    # Parse the XML response
    root = ET.fromstring(response_text)

    # Extract the desired information
    info = {}


    # Extract the identifier
    identifier_element = root.find('.//{http://www.openarchives.org/OAI/2.0/}identifier')
    if identifier_element is not None:
        info['identifier'] = identifier_element.text

    # Extract the datestamp
    datestamp_element = root.find('.//{http://www.openarchives.org/OAI/2.0/}datestamp')
    if datestamp_element is not None:
        info['datestamp'] = datestamp_element.text

    # Extract the set spec
    setspec_element = root.find('.//{http://www.openarchives.org/OAI/2.0/}setSpec')
    if setspec_element is not None:
        info['setSpec'] = setspec_element.text

    # Extract the arXiv ID
    id_element = root.find('.//{http://arxiv.org/OAI/arXiv/}id')
    if id_element is not None:
        info['id'] = id_element.text

    # Extract the created date
    created_element = root.find('.//{http://arxiv.org/OAI/arXiv/}created')
    if created_element is not None:
        info['created'] = created_element.text

    # Extract the updated date
    updated_element = root.find('.//{http://arxiv.org/OAI/arXiv/}updated')
    if updated_element is not None:
        info['updated'] = updated_element.text

    # Extract the authors
    authors = []
    author_elements = root.findall('.//{http://arxiv.org/OAI/arXiv/}author')
    for author_element in author_elements:
        author = {}
        keyname_element = author_element.find('.//{http://arxiv.org/OAI/arXiv/}keyname')
        if keyname_element is not None:
            author['keyname'] = keyname_element.text
        forenames_element = author_element.find('.//{http://arxiv.org/OAI/arXiv/}forenames')
        if forenames_element is not None:
            author['forenames'] = forenames_element.text
        authors.append(author)
    info['authors'] = authors

    # Extract the title
    title_element = root.find('.//{http://arxiv.org/OAI/arXiv/}title')
    if title_element is not None:
        info['title'] = title_element.text

    # Extract the categories
    categories_element = root.find('.//{http://arxiv.org/OAI/arXiv/}categories')
    if categories_element is not None:
        info['categories'] = categories_element.text

    # Extract the license
    license_element = root.find('.//{http://arxiv.org/OAI/arXiv/}license')
    if license_element is not None:
        info['license'] = license_element.text

    # Extract the abstract
    abstract_element = root.find('.//{http://arxiv.org/OAI/arXiv/}abstract')
    if abstract_element is not None:
        info['abstract'] = abstract_element.text

    return info

if __name__ == "__main__":
    # Example usage
    async def main():
        arxiv_id = '2403.07874'  # Replace with the desired arXiv ID
        info = await get_arxiv_info_async(arxiv_id)
        print(info)

    asyncio.run(main())