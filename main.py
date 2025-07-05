import re
import time
import random
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import numpy as np
import joblib
from catboost import CatBoostRegressor
import argparse
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os
import json


def update_progress(current, total, file_path="progress.json"):
    """
        Safely updates a progress JSON file with the current status of a task.

        Args:
            current (int): The current progress count.
            total (int): The total number of items/tasks.
            file_path (str): The path to the JSON file where progress is saved.

        Behavior:
            - Writes progress as JSON to a temporary file (atomic write).
            - Replaces the original progress file with the updated one.
            - Logs errors to stderr if writing fails.

        Example Output (progress.json):
            {
                "current": 34,
                "total": 100,
                "progress_pct": 34
            }
    """
    try:
        progress_data = {
            "current": current,
            "total": total,
            "progress_pct": int((current / total) * 100) if total else 0
        }

        tmp_path = file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(progress_data, f)

        os.replace(tmp_path, file_path)  # Atomic swap
    except Exception as e:
        print(f"Failed to update progress: {e}", file=sys.stderr)
        

def scrape_yad2_from_filtered_url(base_url: str, pages_num: int, delay: float = 1.0) -> pd.DataFrame:
    """
    Scrapes car listing data from Yad2 using Playwright and returns a DataFrame.

    Args:
        base_url (str): The Yad2 search results URL (must include 'page=' parameter).
        pages_num (int): Number of pages to iterate through and scrape listings from.
        delay (float): Time delay (in seconds) between page navigations to reduce detection risk.

    Returns:
        pd.DataFrame: A DataFrame of listings with structured fields.
    """

    listing_urls = []

    # Dictionary to convert Hebrew labels into English keys
    hebrew_to_english = {
        "קילומטראז׳": "mileage", "צבע": "color", "בעלות נוכחית": "ownership",
        "טסט עד": "test_date", "בעלות קודמת": "previous_ownership", "תיבת הילוכים": "transmission",
        "תאריך עליה לכביש": "on_road_date", "סוג מנוע": "fuel_type", "מרכב": "body_type",
        "מושבים": "seats", "כוח סוס": "horsepower", "נפח מנוע": "engine_volume",
        "צריכת דלק משולבת": "fuel_consumption", "סוג הנעה": "drive_type", "מערכת הנעה": "drive_system"
    }

    # Start a Playwright browser session
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir="playwright_user_data",
            headless=False  # Open browser visibly so you can solve CAPTCHA
        )
        page = browser.new_page()

        # Inject anti-bot evasion script before any page loads
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

        # Loop through specified number of pages and collect listing URLs
        for page_num in range(1, pages_num + 1):
            page_url = re.sub(r"page=\d+", f"page={page_num}", base_url)
            page.goto(page_url)
            page.wait_for_selector("a[href^='item/']", timeout=30000)
            links = page.eval_on_selector_all("a[href^='item/']", "els => els.map(e => e.href)")
            listing_urls.extend(links)
            time.sleep(random.uniform(0.5, delay + 0.5))

        listing_urls = list(set(listing_urls))

        total = len(listing_urls)
        all_details = []

        for idx, url in enumerate(listing_urls):
            full_url = f"https://www.yad2.co.il/{url}" if url.startswith("item/") else url

            # progress feedback to the main thread through json
            update_progress(current=idx + 1, total=total)

            try:
                page.goto(full_url)
                page.wait_for_selector("dd", timeout=10000)
                html = page.content()
                soup = BeautifulSoup(html, "html.parser")
                details = {}

                ad_id_tag = soup.find("div", class_="report-ad_adNumber__b1TZP")
                details["listing_id"] = ad_id_tag.text.strip() if ad_id_tag else None

                upload_date_span = soup.find("span", class_="report-ad_createdAt__MhAb0")
                details["upload_date"] = upload_date_span.text.replace("פורסם ב", "").strip() if upload_date_span else None

                details["scrape_date"] = time.strftime("%d/%m/%Y")

                price_span = soup.find("span", attrs={"data-testid": "price"})
                details["price"] = price_span.text.strip().replace(',', '').replace('₪', '') if price_span else None

                title_tag = soup.find("h1")
                details["title"] = title_tag.text.strip() if title_tag else None

                summary_values = page.locator(".details-item_itemValue__r0R14").all()
                details["year_summary"] = summary_values[0].inner_text().strip() if len(summary_values) > 0 else None
                details["owner_count"] = summary_values[1].inner_text().strip() if len(summary_values) > 1 else None

                # Extract additional vehicle details using label-value pairs
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

                # Ensure all expected fields are in the record (even if None)
                for key in hebrew_to_english.values():
                    if key not in details:
                        details[key] = None

                details["url"] = full_url

                if details.get("price"):  # only keep listings with price
                    all_details.append(details)
                # all_details.append(details)
                time.sleep(random.uniform(0.5, 1))

            except Exception as e:
                print(f"Failed to scrape {full_url}: {e}", file=sys.stderr)

        browser.close()

    # Convert to DataFrame and set listing ID as index
    df = pd.DataFrame(all_details)
    df.set_index("listing_id", inplace=True, drop=False)
    return df


def calculate_months_on_road(row):
    if pd.isnull(row['upload_date']) or pd.isnull(row['on_road_date']):
        return None
    delta = (row['upload_date'].year - row['on_road_date'].year) * 12
    delta += row['upload_date'].month - row['on_road_date'].month
    return delta


def parse_on_road(row):
    if pd.notnull(row['on_road_date']):
        return pd.to_datetime(f"01/{row['on_road_date']}", format='%d/%m/%Y', errors='coerce')
    elif pd.notnull(row['year_summary']):
        return pd.to_datetime(f"01/06/{row['year_summary']}", format='%d/%m/%Y', errors='coerce')
    return pd.NaT


def drop_price_outliers_by_title(df, price_col='price', ratio_thresh=10):
    df = df.copy()

    if 'avg_price_by_title' not in df.columns:
        raise ValueError("Expected column 'avg_price_by_title' not found in DataFrame.")

    # Calculate ratio of avg to actual price
    ratio = df['avg_price_by_title'] / df[price_col]

    # Keep rows within threshold range
    mask = (ratio < ratio_thresh) & (ratio > (1 / ratio_thresh))
    df = df[mask].copy()

    return df


def preprocess_car_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Step 0: Drop duplicate listings
    df = df.drop_duplicates(subset='listing_id')
    df = df.reset_index(drop=True)

    # Step 1: Clean price
    df['price'] = df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])

    # Step 2: Clean mileage and engine_volume
    df['mileage'] = df['mileage'].astype(str).str.replace(',', '').str.strip()
    df['engine_volume'] = df['engine_volume'].astype(str).str.replace(',', '').str.strip()

    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df['engine_volume'] = pd.to_numeric(df['engine_volume'], errors='coerce')

    # Step 3: Parse dates
    df['upload_date'] = pd.to_datetime(df['upload_date'], format='%d/%m/%y', errors='coerce')
    df['test_date'] = pd.to_datetime(df['test_date'], errors='coerce')

    # Step 4: Parse on_road_date using external function
    df['on_road_date'] = df.apply(parse_on_road, axis=1)

    # Step 5: Calculate months on road using external function
    df['months_on_road'] = df.apply(calculate_months_on_road, axis=1)

    # Step 6: Extract year and month from on_road_date
    df['on_road_year'] = df['on_road_date'].dt.year
    df['on_road_month'] = df['on_road_date'].dt.month

    # Step 7: Calculate months to next test
    df['months_to_test'] = (
        (df['test_date'].dt.year - df['upload_date'].dt.year) * 12 +
        (df['test_date'].dt.month - df['upload_date'].dt.month)
    )
    df['months_to_test'] = df['months_to_test'].clip(lower=0)

    # Step 8: Join with title_aggregates and drop unknown titles
    try:
        title_ref = pd.read_csv("C:\\Users\\moran\\personal_projects\\vfm_car_finder\\VFM_car_finder\\title_aggregates.csv")
        df = df.merge(title_ref, on="title", how="left")
        df = df.dropna(subset=["avg_price_by_title"])
    except Exception:
        print("No titles found")
        pass

    # Step 9: Drop price outliers based on title-level averages
    df = drop_price_outliers_by_title(df)

    # Step 10: Feature engineering - ratios to title averages
    df['mileage_vs_avg_title'] = df['mileage'] / df['avg_mileage_by_title'].replace(0, np.nan)
    df['months_vs_avg'] = df['months_on_road'] / df['avg_months_on_road_by_title'].replace(0, np.nan)

    return df


def predict_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Load preprocessing components and features
    preprocessor, encoder, features, num_features, cat_features = joblib.load("lgbm_preprocessor.joblib")

    # Extract and clean the input features
    X = df[features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Coerce object-type numerics to numeric
    for col in X.columns:
        if X[col].dtype == 'object' and col not in cat_features:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Impute missing values
    X_array = preprocessor.transform(X)
    X_imputed = pd.DataFrame(X_array, index=X.index)
    X_imputed.columns = (num_features + cat_features)[:X_imputed.shape[1]]

    # Ensure numeric columns are numeric
    for col in num_features:
        if col in X_imputed.columns:
            X_imputed[col] = pd.to_numeric(X_imputed[col], errors='coerce')

    # Encode categorical features
    cat_cols = [col for col in X_imputed.columns if col in cat_features]
    X_imputed[cat_cols] = encoder.transform(X_imputed[cat_cols])

    # Load model and predict
    model = joblib.load("final_lgbm_model.pkl")
    df['predicted_price'] = model.predict(X_imputed)

    return df


def add_price_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Percentage difference between predicted and actual price
    df['price_diff_pct'] = 100 * (df['price'] - df['predicted_price']) / df['predicted_price']

    # Calculate a relative value-for-money score based on price deviation and model uncertainty
    if 'std_error_pct' in df.columns:
        df['price_diff_vs_error'] = df['price_diff_pct'] / df['std_error_pct']
    else:
        df['price_diff_vs_error'] = None  # fallback if std_error_pct is missing

    return df


def vfm_menu(df: pd.DataFrame):
    """
       Developer utility menu for interactively exploring top Value-for-Money (VFM) car listings.

       This CLI tool is intended for debugging, QA, or exploratory analysis by developers,
       separate from the end-user application interface.

       Args:
           df (pd.DataFrame): A DataFrame of car listings that includes at least the following columns:
               - 'title': Car title or name.
               - 'price_diff_vs_error': VFM score based on model error vs. actual price deviation.
               - 'url': Link to the listing.

       Behavior:
           - Prompts developer to input how many top VFM listings they want to view.
           - Sorts listings by ascending 'price_diff_vs_error' (lower = better deal).
           - Prints out the top N listings with title, % cheaper than expected, and URL.
           - Repeats until the user types 'exit' or 'quit'.

       Notes:
           - Intended for internal CLI debugging, not for production UI.
           - Assumes 'price_diff_vs_error' contains numeric values (can be negative).
    """
    df = df.copy()
    evaluated_count = len(df)
    print(f"\n✅ Evaluated {evaluated_count} listings.\n")

    while True:
        try:
            user_input = input("🔢 How many top VFM cars would you like to see? (or type 'exit' to quit): ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("👋 Exiting menu.")
                break

            n = int(user_input)
            if n <= 0:
                print("⚠️ Please enter a positive number.")
                continue

            top_df = df.sort_values(by='price_diff_vs_error').head(n)

            print(f"\n🚗 Top {n} VFM listings:\n")

            for i, row in enumerate(top_df[['title', 'price_diff_vs_error', 'url']].itertuples(index=False), start=1):
                print(f"{i}. {row.title}")
                print(f" {-1*row.price_diff_vs_error:.1f}% cheaper than expected")
                print(f"   🔗 {row.url}\n")

            print("\n🔁 You can try another number or type 'exit' to finish.\n")

        except ValueError:
            print("⚠️ Invalid input. Please enter a number or 'exit'.")

def main(base_url: str, pages_num: int):
    df = scrape_yad2_from_filtered_url(base_url, pages_num)
    df = preprocess_car_data(df)
    df = predict_prices(df)
    df = add_price_diff_features(df)
    vfm_menu(df)


if __name__ == "__main__":
    print("VFM car finder")


    parser = argparse.ArgumentParser(description="Run Yad2 full pipeline: scrape, process, predict, rank")
    parser.add_argument("--url", type=str, required=True, help="Filtered Yad2 search URL (must include page=1)")
    parser.add_argument("--pages", type=int, required=True, help="Number of pages to scrape and evaluate")

    args = parser.parse_args()
    print("Running VFM pipeline with arguments:")
    print("URL:", args.url)
    print("Pages:", args.pages)

    main(base_url=args.url, pages_num=args.pages)
