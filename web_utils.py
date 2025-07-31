import requests
from bs4 import BeautifulSoup

def fetch_website_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        text_content = ' '.join([element.get_text().strip() for element in paragraphs])
        
        return text_content if text_content else None
        
    except Exception as e:
        raise Exception(f"Error fetching website content: {str(e)}")

def process_content(text, max_length=28000):
    cleaned_text = ' '.join(text.split())
    return cleaned_text[:max_length] if len(cleaned_text) > max_length else cleaned_text







