import subprocess

def run_step(description, script_path):
    print (f"Running step: {description}")
    result = subprocess.run(["python3", script_path])
    if result.returncode != 0:
        print(f"Error in:{description}")
        exit(1)
    print(f"Finished: {description}")


if __name__ == "__main__":
    print("Starting Earthquake Data Processing Pipeline")

    run_step("Preprocessing raw data", "scripts/preprocess.py")
    run_step("Importing cleaned data into database", "scripts/import_data.py")
    run_step("Feature Engineering", "scripts/feature_engineering.py")

    print("\n Pipline completed successfully!")
