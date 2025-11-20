import os
import pandas as pd

# Ensure output directory exists
output_dir = "Articles"
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# Load CSV
csv_path = "Articles.csv"
try:
	df = pd.read_csv(csv_path, encoding="latin1")
except Exception as e:
	print(f"Error loading CSV: {e}")
	df = pd.DataFrame()

# Write each row to a separate text file
for idx, row in df.iterrows():
	filename = os.path.join(output_dir, f"article_{idx+1}.txt")
	content = f"Heading: {row.get('Heading', '')}\n\nArticle: {row.get('Article', '').strip()}\n\nDate: {row.get('Date', '')}\nNewsType: {row.get('NewsType', '')}"
	with open(filename, "w", encoding="utf-8") as f:
		f.write(content)
