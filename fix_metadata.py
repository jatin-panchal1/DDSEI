import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load your database credentials
load_dotenv()
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

# The exact 4 IDs your orchestrator just pulled metrics for
data = {
    "content_id": [
        "dQw4w9WgXcQ", # Rickroll
        "jNQXAC9IVRw", # Me at the zoo
        "DXPfCfQj3Up", # IG Shortcode 1
        "DVI5fHsDDvg"  # IG Shortcode 2
    ],
    "duration_seconds": [212, 19, 45, 60],
    "content_type": ["Video", "Video", "Short", "Short"]
}

# Create a dataframe and append it to your metadata table
df = pd.DataFrame(data)

try:
    df.to_sql("content_metadata", con=engine, if_exists="append", index=False)
    print("✅ Successfully injected missing metadata into the database!")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    engine.dispose()