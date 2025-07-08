# 🚗 Best Value-for-Money Used Car Finder (Israel Edition) 🇮🇱  
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange)  
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)  
![ML Model](https://img.shields.io/badge/Model-LightGBM-success)  
![Web Scraping](https://img.shields.io/badge/Scraping-Playwright-green)  
![License](https://img.shields.io/badge/Data%20Access-Restricted-red)

---

## 🎯 Project Overview

This project helps users **find the best value-for-money (VFM) used cars** in Israel, based on current listings on [Yad2.co.il](https://www.yad2.co.il), the country's largest second-hand marketplace.

It combines:

- 🕸️ Real-time web scraping
- 🤖 Machine learning-based price prediction
- 📉 Smart scoring logic based on **standardized error**
- 🌐 An interactive Streamlit app for users to analyze cars live

---

## 🔁 Project Workflow & Real-Time Pipeline

This project was developed in **two phases**: offline training, then online deployment.

### 📦 Phase 1: Data Collection & Model Training (Offline)

- ✅ Scraped **over 40,000 car listings** using Playwright-based scraper
- 🧼 **Cleaned & engineered features**:
  - Normalized Hebrew labels
  - Cleaned numeric fields: price, mileage, engine volume
  - Parsed and aligned multiple date fields
  - Created features like `months_on_road`, `mileage_vs_avg_title`
  - Removed outliers based on **title-level price distributions**
- 🤖 **Trained a LightGBM regressor** to estimate market price
  - Used hybrid numeric/categorical features
  - Validated with holdout set and MAE
- 📏 **Implemented category-level error estimation**:
  - Calculated **standard deviation of residuals per car title**
  - Used as a proxy for model confidence
  - Applied to **scale price deviations** into meaningful VFM scores:
    **VFM Score = (Actual Price - Predicted Price) / Std(Error for that Title)**
- 🔐 Saved the model and preprocessing pipeline for deployment

### 🌐 Phase 2: Real-Time VFM Evaluation Pipeline (Online)

- 🧠 Built an end-to-end **live prediction system** that:
  1. Accepts a user-provided **filtered Yad2 URL**
  2. Launches a scraper via **subprocess** with progress tracking
  3. Cleans and preprocesses new listings
  4. Predicts each car's **fair price** using the trained model
  5. Calculates a **VFM score** (price deviation normalized by expected error)
  6. Labels listings with intuitive tags:
     - 🔥 **Good Deal**
     - ✅ **Fair Price**
     - 💸 **Overpriced**
- 📊 Results are shown in an **interactive Streamlit dashboard**, with:
  - Progress bar + ETA during scraping
  - Sortable, styled table of top listings
  - Direct links to each car ad

---

## 🧠 Architecture Overview

Yad2 Filtered URL
│
▼
[🧭 Playwright Scraper] (real-time subprocess)
│
▼
[🧼 Cleaning + Feature Engineering]
│
▼
[🤖 Trained LightGBM Regressor]
│
▼
[📈 Price Prediction + VFM Score (Normalized by Std)]
│
▼
[🌐 Streamlit UI]

---

## 🛠️ Tech Stack

| Layer        | Tools/Frameworks                                |
|--------------|--------------------------------------------------|
| Scraping     | Python, Playwright, BeautifulSoup                |
| Data         | Pandas, NumPy, joblib                            |
| Modeling     | LightGBM, CatBoost, Scikit-learn                 |
| Visualization| Matplotlib, Seaborn (EDA phase)                  |
| App / UI     | Streamlit                                        |
| Utilities    | argparse, subprocess, JSON, Regex                |

---

## 🚀 How to Run the Project

> ⚠️ Python 3.10+ is required. Chrome must be installed locally for Playwright.

### 🔧 1. Clone & Install Requirements
```bash
git clone https://github.com/yourusername/vfm-used-car-finder.git
cd vfm-used-car-finder
pip install -r requirements.txt

🧪 2. Run the Full CLI Pipeline
python main.py --url "<filtered Yad2 URL with ?page=1>" --pages 3

🌐 3. Launch the Streamlit App
streamlit run app.py
Go to http://localhost:8501 and paste a filtered search URL to begin.

🟡 Important:
Before pasting a URL, make sure you first filter the car listings on Yad2.co.il using their built-in search engine.
Apply filters like:

Manufacturer (e.g., Toyota)
Model (e.g., Corolla)
Year range
Price range
Engine type, etc.

Then copy the filtered results URL, which should include parameters like manufacturer=, model=, and page=1.

✅ Example valid URL:
https://www.yad2.co.il/vehicles/private-cars?manufacturer=mazda&model=3&page=1

🚫 Avoid copying unfiltered or generic pages like:
https://www.yad2.co.il/vehicles/private-cars


📋 Example Use Case
Looking for a Toyota Corolla?
Paste this (example) URL in the app:
https://www.yad2.co.il/vehicles/private-cars?manufacturer=toyota&model=corolla&page=1
You'll get:

🔍 Live scraping + price prediction

📊 VFM scores

✅ Deal recommendations

📦 Repo Structure
.
├── app.py                 # Streamlit app
├── main.py                # End-to-end pipeline (scrape → predict → score)
├── scrape_runner.py       # Subprocess scraper wrapper
├── web_scrapper.py        # Manual alternative scraper
├── data_preprocess.ipynb  # EDA + cleaning
├── model_train.ipynb      # Model training + error estimation
├── final_lgbm_model.pkl   # Trained LightGBM model
├── lgbm_preprocessor.joblib  # Preprocessing pipeline
└── requirements.txt
🔒 Data Access
Due to scraping policies and licensing restrictions, the full dataset (~40,000 records) is not published.

📩 If you're a recruiter, hiring manager, or collaborator, you’re welcome to contact me for a live walkthrough or sample.

💼 Skills Demonstrated
✅ Full ML lifecycle: scraping, cleaning, modeling, deployment

🕵️‍♂️ Advanced scraping with Playwright + CAPTCHA resilience

🧠 Price modeling using LightGBM and engineered features

📏 Error-aware ranking via category-based standard deviation

🧪 Real-world, explainable recommendation logic

🌐 Clean, live user interface with Streamlit

✨ Possible Future Enhancements
🧮 Add confidence intervals to predictions

🎯 Let users filter by budget, year, fuel type

☁️ Deploy to Streamlit Cloud or AWS Lambda

📱 Optimize UI for mobile users

🤝 Let's Connect
📬 Want to explore the app or learn more?
Feel free to reach out:

LinkedIn: https://www.linkedin.com/in/moranshavit/

Email: moranshavit6@gmail.com

“Making car shopping smarter — one ML prediction at a time.”

---
