# 🚀 Unlox: Omni-Channel Social Engagement Engine

Unlox is a full-stack, AI-powered social media analytics platform designed to extract, analyze, and predict content performance across YouTube and Instagram. 

Built entirely in Python, it combines automated data scraping, NLP sentiment analysis, and machine learning into a unified Streamlit dashboard, allowing creators and marketers to make data-driven decisions.

## ✨ Core Features

* **📥 Omni-Channel Data Extractor:** Paste any YouTube link or Instagram shortcode. The backend ETL pipeline automatically scrapes live engagement metrics, fetches comments, runs NLP sentiment analysis to calculate a "Relatable Rate," and upserts everything into a MySQL database.
* **🤖 Virality Predictor:** Features a custom-trained Machine Learning model (`RandomForestRegressor`) that analyzes historical database metrics (views, likes, shares, duration) to predict the "Viral Coefficient" of new, unreleased content.
* **🧪 A/B Testing Lab:** A built-in statistical analysis tool that runs two-proportion Z-tests on early content hooks to determine statistical significance and declare winning variants with 95% confidence intervals.
* **📊 PowerBI Dashboard Integration:** Seamless integration with embedded PowerBI reports to visualize database growth and audience sentiment over time.

## 💻 Tech Stack

* **Frontend / UI:** Streamlit
* **Backend Pipeline:** Python, Pandas, Regex
* **Database:** MySQL, SQLAlchemy
* **Machine Learning:** Scikit-Learn (`RandomForestRegressor`), Joblib
* **Statistical Analysis:** Statsmodels (Two-proportion Z-testing)
* **Environment Management:** Python-dotenv

## ⚙️ Installation & Setup

Follow these steps to run Unlox on your local machine.

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/unlox.git](https://github.com/yourusername/unlox.git)
cd unlox
```
2. Install dependencies
Make sure you have Python installed, then run:

Bash
```
pip install -r requirements.txt
```
(Requires: streamlit, pandas, scikit-learn, statsmodels, sqlalchemy, pymysql, python-dotenv, joblib)

3. Configure Environment Variables
Create a .env file in the root directory and add your MySQL database connection string and PowerBI embed URL:

Code snippet
DATABASE_URL="mysql+pymysql://username:password@localhost/unlox_db"
POWERBI_EMBED_URL="[https://app.powerbi.com/reportEmbed?reportId=](https://app.powerbi.com/reportEmbed?reportId=)..."
4. Train the Virality AI Model
Before running the dashboard, train the Random Forest model on your existing database metrics:

Bash
python virality_model.py
(Note: This will generate a virality_model.joblib file inside a models/ directory).

5. Launch the Dashboard
Start the Streamlit web application:

Bash
streamlit run app.py
The application will automatically open in your browser at http://localhost:8501.
