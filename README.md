# nba-player-similarity-explorer
A Flask and machine-learning web app for exploring similar NBA players, with roster recommendations planned for a future update.

## Run locally

This project uses Python, Flask, pandas, scikit-learn, and `nba_api`. Run these commands from the project folder in PowerShell.

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Download the 2025–26 regular season player statistics and prepare the dataset:

   ```powershell
   python scripts/fetch_player_stats.py
   python ml/preprocess.py
   ```

4. Start the application:

   ```powershell
   python app.py
   ```

5. Open http://127.0.0.1:5000 in your browser. Enter a player’s name, such as Stephen Curry, and select **Find similar players**. Press **Ctrl+C** in the terminal when you are finished.

The downloaded and processed CSV files are generated locally and are excluded from Git. Run step 3 after cloning the repository.