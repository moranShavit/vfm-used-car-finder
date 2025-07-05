import streamlit as st
import pandas as pd
import json
import os
import subprocess
import time
from main import (
    preprocess_car_data,
    predict_prices,
    add_price_diff_features
)


def read_progress(file_path="progress.json"):
    """
       Reads the progress JSON written by the scraper subprocess.

       Returns:
           dict or None: JSON object with keys like 'current', 'total', 'progress_pct',
                         or None if file doesn't exist or is malformed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def run_scraper_subprocess(base_url, pages_num) -> pd.DataFrame:
    """
    Executes the scraper in a separate Python process and tracks its progress.

    Args:
        base_url (str): Yad2 filtered search URL (must include a page parameter).
        pages_num (int): Number of pages to scrape.

    Returns:
        pd.DataFrame: Listings as a DataFrame, or empty if scraping failed.
    """
    try:
        # Launch subprocess to run the scraper (must be defined in scrape_runner.py)
        process = subprocess.Popen(
            ['python', 'scrape_runner.py', base_url, str(pages_num)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # --- Progress bar in Streamlit ---
        progress_bar = st.progress(0)
        status = st.empty()
        start_time = time.time()
        pct = 0

        # Monitor progress.json until subprocess completes
        while process.poll() is None and pct < 100:
            progress = read_progress()
            if progress:
                pct = progress.get("progress_pct", 0)
                done = progress.get("current", 0)
                total = progress.get("total", 1)

                elapsed = time.time() - start_time
                eta = (elapsed / done * (total - done)) if done > 0 else 0
                eta_str = f"ETA: ~{int(eta // 60)}:{int(eta % 60):02d}"

                progress_bar.progress(pct)
                status.text(f"Scraping {done}/{total} listings... {eta_str}")
            time.sleep(10)

        # Clean up progress tracking file
        if os.path.exists("progress.json"):
            os.remove("progress.json")

        # Read output and error from the scraper subprocess
        stdout_bytes, stderr_bytes = process.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="ignore")
        stderr = stderr_bytes.decode("utf-8", errors="ignore")

        if process.returncode != 0:
            st.error("❌ Scraper subprocess failed.")
            st.text(stderr[:500])
            return pd.DataFrame()

        # Parse stdout as JSON and convert to DataFrame
        try:
            listings = json.loads(stdout)
            return pd.DataFrame(listings)
        except json.JSONDecodeError:
            st.error("❌ Could not decode output from scraper.")
            st.text(stdout[:500])  # Preview the start of stdout



            return pd.DataFrame()

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return pd.DataFrame()



def evaluate_multiple_listings(base_url: str, pages_num: int) -> pd.DataFrame:
    """
    Scrapes listings from Yad2, processes them, and evaluates value-for-money.

    Steps:
        1. Run the scraper subprocess.
        2. Preprocess and clean data.
        3. Predict prices using trained model.
        4. Calculate VFM score and label each listing.

    Returns:
        pd.DataFrame: Final evaluated dataset with VFM scores and recommendations.
    """
    try:
        df = run_scraper_subprocess(base_url, pages_num)
        df = preprocess_car_data(df)
        df = predict_prices(df)
        df = add_price_diff_features(df)

        if df.empty:
            st.warning("No valid listings were found or could be processed.")
            return pd.DataFrame()

        # Label each listing based on how good of a deal it is
        def label_recommendation(row):
            if pd.isnull(row['price_diff_vs_error']):
                return "❔ Not enough data"
            elif row['price_diff_vs_error'] < -1:
                return "🔥 Good deal"
            elif -1 <= row['price_diff_vs_error'] <= 1:
                return "✅ Fair price"
            else:
                return "💸 Overpriced"

        df["Recommendation"] = df.apply(label_recommendation, axis=1)
        df["VFM Score"] = -1 * df["price_diff_vs_error"]

        return df

    except Exception as e:
        st.error(f"❌ Error while evaluating listings: {str(e)}")
        return pd.DataFrame()


# --- Streamlit UI ---
st.set_page_config(page_title="VFM Used Car Finder", layout="centered")

st.title("🚗 Best Value-for-Money Used Cars in Israel")
st.markdown("Paste a **filtered Yad2 URL** and select how many pages to evaluate:")

# Input fields for URL and number of pages
url_input = st.text_input("🔗 Filtered Search URL", placeholder="https://www.yad2.co.il/vehicles/private-cars?manufacturer=...")
pages_input = st.number_input("📄 Number of Pages to Evaluate", min_value=1, max_value=50, value=1, step=1)

# Evaluate button
if st.button("🔍 Evaluate Listings"):
    if not url_input.strip():
        st.warning("Please enter a valid Yad2 filtered search URL.")
    else:
        with st.spinner("Scraping and evaluating listings..."):
            results_df = evaluate_multiple_listings(url_input, pages_input)

        if not results_df.empty:
            st.success(f"✅ Found and evaluated {len(results_df)} listings.")

            # Sort listings by VFM Score
            top_df = results_df.sort_values(by="VFM Score", ascending=False).copy()

            # Prepare final display table
            display_df = top_df[[
                "title", "price", "predicted_price", "VFM Score", "Recommendation", "url"
            ]].rename(columns={
                "title": "Title",
                "price": "Listing Price (₪)",
                "predicted_price": "Predicted Price (₪)",
                "url": "Listing URL"
            })

            # Show results in a styled Streamlit table
            st.markdown("### 📋 Top VFM Listings")
            st.dataframe(display_df.style.format({
                "Listing Price (₪)": "₪{:.0f}",
                "Predicted Price (₪)": "₪{:.0f}",
                "VFM Score": "{:.2f}"
            }), use_container_width=True)

            st.markdown("📥 Click on a link in the 'Listing URL' column to view the car.")


