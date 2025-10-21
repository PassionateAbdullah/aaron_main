import pandas as pd
from typing import Dict, Any
import json


def load_invoice_preview(path: str, n: int = 10, save_preview: bool = False, preview_path: str = "invoice_preview.csv") -> Dict[str, Any]:
	"""Load invoice CSV and return a preview and basic statistics.

	Args:
		path: Full path to the CSV file.
		n: Number of rows to include in the head preview.
		save_preview: If True, saves the first n rows to `preview_path`.
		preview_path: Path to save the preview CSV when save_preview is True.

	Returns:
		A dict with keys: shape, columns, head (list of records), dtypes, null_counts, unique_counts.
	"""
	# Only load the requested columns to save memory and focus on the task
	usecols = ["case_id", "activity_name", "start_time", "complete_time"]
	df = pd.read_csv(path, usecols=usecols)

	# Parse datetime-like columns where possible
	for col in ("start_time", "complete_time"):
		try:
			df[col] = pd.to_datetime(df[col], errors="coerce")
		except Exception:
			# Leave as-is if parsing fails
			pass

	result = {
		"shape": df.shape,
		"columns": df.columns.tolist(),
		"head": df.head(n).to_dict(orient="records"),
		"dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
		"null_counts": df.isnull().sum().to_dict(),
		"unique_counts": df.nunique(dropna=False).to_dict(),
	}

	if save_preview:
		df.head(n).to_csv(preview_path, index=False)

	return result


def process_invoice_data(path: str) -> pd.DataFrame:
	"""
	Reads invoice data, calculates durations, and determines the ideal position for each activity.

	Args:
		path: Full path to the CSV file.

	Returns:
		A new pandas DataFrame with the following added columns:
		- 'duration_seconds': The difference between complete_time and start_time in seconds.
		- 'step_in_case': The chronological step number of an activity within its case.
		- 'ideal_position': The most common step number (mode) for each activity type across all cases.
	"""
	# 1. Read CSV and select the first four columns
	usecols = ["case_id", "activity_name", "start_time", "complete_time"]
	df = pd.read_csv(path, usecols=usecols)

	# Convert time columns to datetime objects
	df["start_time"] = pd.to_datetime(df["start_time"])
	df["complete_time"] = pd.to_datetime(df["complete_time"])

	# 2. Create a new column for overall completed time in seconds
	df["duration_seconds"] = (df["complete_time"] - df["start_time"]).dt.total_seconds()

	# 3. Determine the ideal position for every activity
	# First, sort by case and time to establish the correct order of steps
	df = df.sort_values(by=["case_id", "start_time"]).reset_index(drop=True)

	# Find the step number for each activity within its case
	df["step_in_case"] = df.groupby("case_id").cumcount() + 1

	# Now, find the most frequent step number (mode) for each unique activity name.
	# This will be the "ideal position".
	# .mode()[0] is used to get the first mode if there are multiple.
	ideal_position_map = df.groupby("activity_name")["step_in_case"].apply(lambda x: x.mode()[0])

	# Create the new 'ideal_position' column by mapping the activity name to its ideal position
	df["ideal_position"] = df["activity_name"].map(ideal_position_map)

	# 4. Return the new dataset (DataFrame) with the changes
	return df


if __name__ == "__main__":
	# Quick demo when running this file directly
	import os

	workspace_path = os.path.dirname(__file__)
	csv_path = os.path.join(workspace_path, "invoice_process_expanded (1).csv")

	try:
		info = load_invoice_preview(csv_path, n=10, save_preview=False)
		print("shape:", info["shape"])
		print("columns:", info["columns"])
		if info["head"]:
			print("first row:")
			for k, v in info["head"][0].items():
				print(f"  {k}: {v}")
	except FileNotFoundError:
		print(f"CSV not found at: {csv_path}")
	except Exception as e:
		print("Error loading CSV:", e)

	print("\n--- Running New Processing Function ---")
	try:
		# Process the data to get the new DataFrame
		processed_df = process_invoice_data(csv_path)

		# Display the first 15 rows of the result as a JSON string for readability
		print("Sample of processed data with new columns:")
		print(processed_df.head(15).to_json(orient="records", indent=4, date_format="iso"))
	except Exception as e:
		print(f"Error processing data: {e}")
