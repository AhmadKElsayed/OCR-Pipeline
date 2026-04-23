# Import the massive evaluation function from your other file
from battleground import run_battleground

if __name__ == "__main__":
    TARGET_DIRECTORY = "test_data"
    
    OUTPUT_DIRECTORY = "battleground_results"

    print("🚀 Initializing OCR Battleground Pipeline...")
    
    # Run the imported function
    run_battleground(
        test_dir=TARGET_DIRECTORY, 
        output_dir=OUTPUT_DIRECTORY
    )
    
    print("✅ Run complete. You can safely close the server.")