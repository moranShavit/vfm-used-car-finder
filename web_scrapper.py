import os

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import re
import random
import time

def scrape_yad2_listings(start_page, end_page, delay=2):
    """
        Scrapes car listings from the main Yad2 cars page (https://www.yad2.co.il/vehicles/cars)
        and returns the structured data in a pandas DataFrame.

        This function uses Playwright to automate browser actions, bypass simple bot detection,
        and navigate multiple pages of car listings. Each listing is then parsed with BeautifulSoup
        to extract detailed vehicle information.

        Args:
            start_page (int): The first page number of listings to scrape.
            end_page (int): The last page number of listings to scrape.
            delay (int): Seconds to wait between individual listing requests to avoid detection.

        Returns:
            pd.DataFrame: DataFrame containing detailed information for each unique car listing.
    """
    listing_urls = []

    # Step 1: Collect listing URLs with Playwright
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir="playwright_user_data",
            headless=False  # Open browser visibly so you can solve CAPTCHA
        )
        page = browser.new_page()

        # Inject anti-bot JavaScript to spoof navigator properties
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            Object.defineProperty(navigator, 'languages', {
                get: () => ['he-IL', 'en-US']
            });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4]
            });
        """)

        # Loop over pages to collect all ad URLs
        for page_num in range(start_page, end_page + 1):
            print(f"Loading page {page_num}")
            page.goto(f"https://www.yad2.co.il/vehicles/cars?priceOnly=1&Order=1&page={page_num}")

            # Wait for car listings (ads) to be visible before scraping links
            page.wait_for_selector("a[href^='item/']", timeout=30000)

            # Extract hrefs of all listing links on the page
            ad_links = page.eval_on_selector_all(
                "a[href^='item/']",
                "elements => elements.map(e => e.href)"
            )
            listing_urls.extend(ad_links)
            time.sleep(random.uniform(0.5, 2))

        # Remove duplicate links (just in case of overlap or duplicates on the site)
        listing_urls = list(set(listing_urls))  # Remove duplicates
        print(f"Found {len(listing_urls)} unique listings.")

        # Step 2: Visit each listing page and extract detailed info
        all_details = []
        headers = {"User-Agent": "Mozilla/5.0"}

        # Hebrew field names mapped to internal English keys
        hebrew_to_english = {
            "קילומטראז׳": "mileage",
            "צבע": "color",
            "בעלות נוכחית": "ownership",
            "טסט עד": "test_date",
            "בעלות קודמת": "previous_ownership",
            "תיבת הילוכים": "transmission",
            "תאריך עליה לכביש": "on_road_date",
            "סוג מנוע": "fuel_type",
            "מרכב": "body_type",
            "מושבים": "seats",
            "כוח סוס": "horsepower",
            "נפח מנוע": "engine_volume",
            "צריכת דלק משולבת": "fuel_consumption",
            "סוג הנעה": "drive_type",
            "מערכת הנעה": "drive_system"
        }

        count = 0
        for url in listing_urls:
            count += 1
            if count%20 == 0:
                print(f"{int(count/len(listing_urls)*100)}% complete")
            full_url = f"https://www.yad2.co.il/{url}" if url.startswith("item/") else url

            try:
                page.goto(full_url)

                page.wait_for_selector("dd", timeout=10000)
                details = {}

                # Parse HTML using BeautifulSoup for static content
                html = page.content()
                soup = BeautifulSoup(html, "html.parser")

                # Get listing ID from hidden element
                ad_id_tag = soup.find("div", class_="report-ad_adNumber__b1TZP")
                details["listing_id"] = ad_id_tag.text.strip() if ad_id_tag else None

                # Extract listing upload date
                try:
                    upload_date_span = soup.find("span", class_="report-ad_createdAt__MhAb0")
                    if upload_date_span:
                        text = upload_date_span.text.replace("\u200d", "").replace("פורסם ב", "").strip()
                        details["upload_date"] = text
                    else:
                        details["upload_date"] = None
                except:
                    details["upload_date"] = None

                # Add current scrape date
                details["scrape_date"] = time.strftime("%d/%m/%Y")

                # Extract price
                try:
                    price_span = soup.find("span", attrs={"data-testid": "price"})
                    details["price"] = price_span.text.strip().replace(',', '').replace('₪', '') if price_span else None
                except Exception as e:
                    details["price"] = None
                    # Log scraping error to file
                    with open("scrape_log.txt", "a", encoding="utf-8") as log:
                        log.write(f"Failed to extract price for {details.get('listing_id', 'unknown')}: {e}\n")

                # Title (make & model)
                title_tag = soup.find("h1")
                if title_tag:
                    details["title"] = title_tag.text.strip()

                # Extract summary fields (year, ownership count, mileage summary)
                summary_values = page.locator(".details-item_itemValue__r0R14").all()
                if len(summary_values) >= 2:
                    details["year_summary"] = summary_values[0].inner_text().strip()
                    details["owner_count"] = summary_values[1].inner_text().strip()
                else:
                    details["year_summary"] = None
                    details["owner_count"] = None

                # Extract all field pairs (label-value) from listing
                for label_elem in page.locator("dd").all():
                    label_text = label_elem.inner_text().strip()
                    key = hebrew_to_english.get(label_text)
                    if key:
                        try:
                            value_elem = label_elem.evaluate_handle("el => el.nextElementSibling")
                            value_text = value_elem.inner_text().strip() if value_elem else None
                            details[key] = value_text
                        except:
                            details[key] = None

                # Fill missing fields with None to ensure uniform structure
                for key in hebrew_to_english.values():
                    if key not in details:
                        details[key] = None

                # Save source URL
                details["url"] = full_url

                all_details.append(details)

                # Delay between listings
                time.sleep(random.uniform(0.5, 1))

                # Longer break every ~77 listings to avoid detection
                if count % 77 == 0:
                    time.sleep(random.uniform(10, 20))  # simulate coffee break 😄


            except Exception as e:
                # Log individual listing scrape failure
                with open("scrape_log.txt", "a", encoding="utf-8") as log:
                    log.write(f"Failed to process {full_url}: {e}\n")

        browser.close()

    # Convert to DataFrame and set index to listing ID
    df = pd.DataFrame(all_details)
    df.set_index("listing_id", inplace=True, drop=False)
    return df


def scrape_and_save(start_page, end_page, output_dir):
    """
        Scrapes Yad2 car listings from a specified page range and saves the result as a CSV file.

        Args:
            start_page (int): The first page to scrape.
            end_page (int): The last page to scrape.
            output_dir (str): Directory where the CSV output will be saved.

        If scraping fails, the error is logged to 'scrape_log.txt' and the function exits.
    """
    try:
        # Run the main scraping function for the page range
        df = scrape_yad2_listings(start_page, end_page)
    except Exception as e:
        # Log failure for this chunk of pages
        with open("scrape_log.txt", "a", encoding="utf-8") as log:
            log.write(f"Failed to scrape pages {start_page}-{end_page}: {e}\n")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build CSV filename using page range
    filename = f"yad2_data_{start_page}_{end_page}_round_2.csv"
    csv_path = os.path.join(output_dir, filename)

    # Save the DataFrame as CSV with UTF-8 BOM for Hebrew compatibility
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Log success to log file
    with open("scrape_log.txt", "a", encoding="utf-8") as log:
        log.write(f"Saved pages {start_page} to {end_page} to {csv_path}\n")

if __name__ == "__main__":
    import argparse
    import os

    # Command-line interface for launching scraping in page chunks
    parser = argparse.ArgumentParser(description="Scrape Yad2 in chunks")
    parser.add_argument("--start", type=int, required=True, help="Start page")
    parser.add_argument("--end", type=int, required=True, help="End page")
    parser.add_argument("--output", type=str, default="scraped_chunks", help="Output directory")
    args = parser.parse_args()
    print("Running scraper with arguments:", args.start, args.end)

    # Break the total page range into chunks of 20 pages for safer scraping
    for chunk_start in range(args.start, args.end + 1, 20):
        chunk_end = min(chunk_start + 19, args.end)
        scrape_and_save(chunk_start, chunk_end, args.output)