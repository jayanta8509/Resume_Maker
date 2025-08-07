import re
import requests
from bs4 import BeautifulSoup
import time
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_portfolio_content(portfolio_link: str, max_retries: int = 3, timeout: int = 10) -> Tuple[Optional[str], Optional[int]]:
    """
    Fetches and processes content from a given portfolio URL.
    
    Args:
        portfolio_link (str): URL of the portfolio website
        max_retries (int): Maximum number of retry attempts
        timeout (int): Request timeout in seconds
        
    Returns:
        Tuple[Optional[str], Optional[int]]: (processed text content, status code)
    """
    # Set headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    for attempt in range(max_retries):
        try:
            # Send GET request to the URL
            response = requests.get(portfolio_link, headers=headers, timeout=timeout)
            
            # Check if request was successful
            if response.status_code == 200:
                # Parse HTML and extract text content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script, style, and other non-content elements
                for element in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
                    element.decompose()
                
                # Get text content and clean it up
                text_content = soup.get_text(separator=' ', strip=True)
                
                # Remove excessive whitespace
                text_content = re.sub(r'\s+', ' ', text_content)
                
                return text_content, response.status_code
            else:
                logger.warning(f"Attempt {attempt + 1}: Received status code {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}: Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
    
    logger.error(f"Failed to fetch content after {max_retries} attempts")
    return None, None