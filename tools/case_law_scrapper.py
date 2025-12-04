import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import time
import re # Added for cleaner sanitization

# --- Configuration ---
BASE_URL = "https://supremecourt.govmu.org/judgment-search"
# The initial search URL for the Industrial Court (Court ID 80507)
# Note: The original URL had an extra, truncated quote; it's cleaned up here.
INITIAL_SEARCH_URL = "https://supremecourt.govmu.org/judgment-search?search_inner=&title=&field_document_number=&jurisdiction=All&delivered_from=&delivered_until=&field_court=80507&summary=&exact_search=true&operator=1&field_delivered_by&from_year=&to_year=&contains_any=&contains_all=&contains_none=&glossaryaz_title=&field_delivered_by_text_long=&page=0"
OUTPUT_DIR = "documents/case_precedent"
DOWNLOAD_DELAY = 1.5
MAX_PAGES = 50 # Set a safe limit

# The list to store all unique case links found across all pages
all_case_links = {}

def create_output_directory(path):
    """Creates the target directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    print(f"Created/Ensured directory: {path}")

def scrape_page(url, page_num):
    """Fetches a single page, extracts file links, and finds the next page link."""
    print(f"Scraping Page {page_num}: {url}")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  --> ERROR: Could not fetch URL {url}. Reason: {e}")
        return None, None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # --- 1. Extract File Links on Current Page ---
    for link in soup.find_all('a', class_='faDownload'):
        raw_url = link['href']
        full_url = urljoin(BASE_URL, raw_url)
          
        title = ""
        # Look for the title link in the same table row, typically the first or second <td>
        parent_row = link.find_parent('tr')
        if parent_row:
             # Look for a common title field class or just the first cell
             title_cell = parent_row.find('td', class_='views-field-title') or parent_row.find('td')
             if title_cell:
                 # Extract the text from the title link or cell content
                 title = title_cell.get_text(strip=True)
                 
        if not title:
             # Fallback: Use a unique ID from the URL as the title
             unique_id = full_url.split('/')[-1]
             title = f"Judgment_File_{unique_id}"
             
        # All download links follow the /downloadPDF/ structure, so we hardcode .pdf
        extension = ".pdf" 
        
        # Store link only if it's new
        if full_url not in all_case_links:
            all_case_links[full_url] = {
                "title": title,
                "url": full_url,
                "extension": extension
            }

    # --- 2. Find Next Page Link (FIXED Pagination Logic) ---
    next_page_url = None
    
    # Method 1: Try the specific class
    next_link_container = soup.find('li', class_='pager__item--next')
    print ('--next->' ,next_link_container)
    
    if next_link_container:
        a_tag = next_link_container.find('a', href=True)
        if a_tag:
            next_page_url = urljoin(BASE_URL, a_tag['href'])
    else:
        print("  --> No 'pager__item--next' found, trying fallback methods.")
    
    # Method 2: Fallback - Search for an anchor tag with the text 'Next' or the right arrow
    if not next_page_url:
        for a_tag in soup.find_all('a', href=True):
            text = a_tag.get_text(strip=True).lower()
            # Check for 'next' or the common right arrow symbol '»'
            if 'next' in text or '»' in text:
                next_page_url = urljoin(BASE_URL, a_tag['href'])
                print("  --> Found next page using text fallback.")
                break 

    if not next_page_url:
        print("  --> No next page found.")
    
    return next_page_url, len(all_case_links)

def download_file(link_data, output_path):
    """Downloads the file from the URL and saves it to the output path."""
    title = link_data['title']
    url = link_data['url']
    extension = link_data['extension']
    
    # --- FIX: Sanitize Filename Correctly ---
    # Use re.sub to keep only alphanumeric, spaces, underscores, and hyphens
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    safe_title = safe_title[:150] # Limit title length
    
    # Use a part of the URL's unique ID as a suffix to ensure uniqueness and prevent empty name
    url_id = url.split('/')[-1]
    
    # Ensure a non-empty safe_title
    if not safe_title:
        safe_title = "Judgment_File"
        
    # Construct the full path, guaranteeing the path and name are separated
    filename = os.path.join(output_path, f"{safe_title}_{url_id}{extension}")
    
    # Create the target directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Skip if the file already exists and is not tiny (e.g., larger than 1KB)
    if os.path.exists(filename) and os.path.getsize(filename) > 1024:
        return 0

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        # Check for HTML content (indicates an error or redirection page)
        content_type = response.headers.get('content-type', '').lower()
        if 'html' in content_type:
             print(f"  -> WARNING: File {url} returned an HTML page (error/redirect). Skipping.")
             return 0

        # Save the content stream
        file_size = 0
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                file_size += len(chunk)
                
        if file_size < 1024: # Check if the downloaded file is suspiciously small
            os.remove(filename) # Remove the tiny file
            print(f"  -> WARNING: Downloaded file {filename} is suspiciously small ({file_size} bytes). Skipping.")
            return 0
                
        print(f"  -> SUCCESS: Downloaded {file_size / 1024:.2f} KB to {filename}")
        return 1

    except Exception as e:
        print(f"  -> ERROR: Failed to download {url}. Reason: {e}")
        # Clean up any partial file that might have been created
        if 'filename' in locals() and os.path.exists(filename):
             os.remove(filename)
        return 0

# --- Main Execution ---
if __name__ == "__main__":
    
    create_output_directory(OUTPUT_DIR)
    
    current_url = INITIAL_SEARCH_URL
    page_counter = 1
    
    print("\n--- Starting Case Law Scraper ---")
    
    while current_url and page_counter <= MAX_PAGES:
        next_url, current_total = scrape_page(current_url, page_counter)
        
        print(f"  --> Total links found so far: {current_total}")
        time.sleep(DOWNLOAD_DELAY) 
        
        current_url = next_url
        page_counter += 1
        
    print(f"\n--- Link Aggregation Complete ---")
    
    unique_case_links = list(all_case_links.values())
    
    if not unique_case_links:
        print("Process finished: No case law file links were found.")
    else:
        print(f"Total unique case law files to download: {len(unique_case_links)}")
        print(f"\nStarting file download into '{OUTPUT_DIR}'...")
        
        successful_downloads = 0
        for link in unique_case_links:
            successful_downloads += download_file(link, OUTPUT_DIR)
            time.sleep(0.5) 
                
        print(f"\n--- Case Law Ingestion Complete ---")
        print(f"Total files successfully downloaded/updated: {successful_downloads} out of {len(unique_case_links)} found.")
        print("The files are ready for text extraction.")