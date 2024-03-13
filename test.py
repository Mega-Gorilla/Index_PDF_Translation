import requests
import xml.etree.ElementTree as ET

def get_license_info(arxiv_id):
    # OAI-PMH endpoint
    base_url = 'http://export.arxiv.org/oai2'
    
    # Set parameters for the OAI-PMH request
    params = {
        'verb': 'GetRecord',
        'identifier': f'oai:arXiv.org:{arxiv_id}',
        'metadataPrefix': 'arXiv',
    }
    
    # Send the OAI-PMH request
    response = requests.get(base_url, params=params)
    
    # Parse the XML response
    root = ET.fromstring(response.text)
    print(ET.tostring(root, encoding='unicode'))
    
    # Find the license information in the metadata
    license_element = root.find('.//{http://arxiv.org/OAI/arXiv/}license')
    
    if license_element is not None:
        license_info = license_element.text
        return license_info
    else:
        return 'License information not found'

# Example usage
arxiv_id = '2402.10949'
license_info = get_license_info(arxiv_id)
print(f'License information for {arxiv_id}: {license_info}')