import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import time

# --- Configuration ---
BASE_URL = "https://pmo.govmu.org"

# FIXED LIST OF TARGET URLS (Main page + 2024 down to 2019)
CABINET_URLS = [
    "https://pmo.govmu.org/Pages/Cabinet_Decisions/Cabinet_Decisions.aspx",
    "https://pmo.govmu.org/Pages/Cabinet_Decisions/Cabinet_Decision2024.aspx",
    "https://pmo.govmu.org/Pages/Cabinet_Decisions/Cabinet_Decisions2023.aspx",
    "https://pmo.govmu.org/Pages/Cabinet_Decisions/Cabinet-Decisions-2022.aspx",
    "https://pmo.govmu.org/Pages/Cabinet_Decisions/Cabinet-Decisions-2021.aspx",
    "https://pmo.govmu.org/Pages/Cabinet_Decisions/Cabinet-Decisions-2020.aspx",
    "https://pmo.govmu.org/Pages/Cabinet_Decisions/Cabinet-Decisions-2019.aspx",
]

# Target directory based on your request
OUTPUT_DIR = "../documents/cabinet_decisions"
DOWNLOAD_DELAY = 1.5 

def create_output_directory(path):
    """Creates the target directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    print(f"Created/Ensured directory: {path}")

def scrape_links_from_url(url):
    """Fetches a single page and extracts relevant decision links."""
    print(f"Scraping page: {url}")
    links = []
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  --> ERROR: Could not fetch URL {url}. Reason: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    
    for link in soup.find_all('a', href=True):
        link_text = link.get_text(strip=True)
        raw_url = link['href']
        full_url = urljoin(BASE_URL, raw_url)
        
        # Filter: Must contain the text AND link to a decision file path
        is_relevant_text = "Highlights of Cabinet Meeting" in link_text
        is_decision_file = "/CabinetDecision/" in full_url or full_url.lower().endswith(('.pdf', '.docx'))
        
        if is_relevant_text and is_decision_file:
            # Determine extension based on URL, defaulting to PDF
            extension = ".pdf" if full_url.lower().endswith(('.pdf', '.docx', '.doc')) else ".pdf"
            
            links.append({
                "title": link_text,
                "url": full_url,
                "extension": extension
            })
            
    # Remove duplicates
    unique_links = list({item['url']: item for item in links}.values())
    print(f"  --> Found {len(unique_links)} unique links on this page.")
    return unique_links

def download_file(link_data, output_path):
    """Downloads the file from the URL and saves it to the output path."""
    title = link_data['title']
    url = link_data['url']
    extension = link_data['extension']
    
    # Create a safe filename using the title
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip().replace(' ', '_')
    filename = os.path.join(output_path, f"{safe_title}{extension}")
    
    if os.path.exists(filename) and os.path.getsize(filename) > 1024:
        # print(f"  -> SKIPPED: File already exists and is complete: {filename}")
        return 0

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        # Save the content stream to the file
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"  -> SUCCESS: Saved file size: {os.path.getsize(filename) / 1024:.2f} KB")
        return 1

    except Exception as e:
        print(f"  -> ERROR: Failed to download {url}. Reason: {e}")
        return 0

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Create Directory
    create_output_directory(OUTPUT_DIR)
    
    all_links = []
    
    # 2. Scrape Fixed List of URLs
    print("\n--- Aggregating links from fixed list of Cabinet Decision pages ---")
    for url in CABINET_URLS:
        # Give a slight pause between accessing different index pages
        time.sleep(DOWNLOAD_DELAY) 
        new_links = scrape_links_from_url(url)
        all_links.extend(new_links)
        
    # Remove duplicates from the combined list
    unique_links = list({item['url']: item for item in all_links}.values())
    
    if not unique_links:
        print("\nProcess finished: No unique links found.")
    else:
        print(f"\nTotal unique decision links scraped: {len(unique_links)}")
        print(f"\nStarting file download into '{OUTPUT_DIR}'...")
        
        successful_downloads = 0
        for link in unique_links:
            successful_downloads += download_file(link, OUTPUT_DIR)
            time.sleep(0.5) # Short pause between file downloads
                
        print(f"\n--- Task 1: Data Ingestion (File Download) Complete ---")
        print(f"Total files successfully downloaded: {successful_downloads} out of {len(unique_links)} unique links found.")
        print(f"The raw files are saved in the '{OUTPUT_DIR}' directory, ready for the ML Filter (Task 2).")