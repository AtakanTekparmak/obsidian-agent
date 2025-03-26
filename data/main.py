import os
from pipeline.steps.generate_stories import get_stories
from pipeline.utils import save_data_to_csv, save_data_to_json
import pandas as pd

def main():
    print("Starting data generation pipeline...")
    
    # Step 1: Generate diverse knowledge base (personal stories)
    print("Step 1: Generating diverse personal stories...")
    stories_df = get_stories(num_stories=5)  # Generating 5 stories for demonstration
    
    # Display summary of generated stories
    print(f"Generated {len(stories_df)} personal stories")
    print("Story subjects:")
    for idx, row in stories_df.iterrows():
        print(f"  - {row['person_name']} (Age: {row['age']})")
    
    # Save stories to CSV and JSON
    csv_path = save_data_to_csv(stories_df, "personal_stories.csv")
    json_path = save_data_to_json(stories_df, "personal_stories.json")
    
    print("Data generation pipeline completed successfully!")
    print(f"Generated data saved to:")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {json_path}")

if __name__ == "__main__":
    main()
