# scrape_runner.py
import sys
from main import scrape_yad2_from_filtered_url

sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    # This script is designed to run the Yad2 scraping process as a separate subprocess.
    # It takes two command-line arguments:
    #   1. base_url: the search URL to scrape (must contain a page parameter like "?page=1")
    #   2. pages_num: the number of result pages to scrape
    # The scraped listings are output as a JSON string to stdout.
    # This allows another Python process (or CLI tool) to capture the output and use it.

    base_url = sys.argv[1]
    pages_num = int(sys.argv[2])

    df = scrape_yad2_from_filtered_url(base_url, pages_num)
    print(df.to_json(orient="records", force_ascii=False))
