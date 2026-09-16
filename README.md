# 🚘 Automatic Number Plate Recognition (ANPR) — Indian Plates

Welcome to my ANPR mini-project! This system detects vehicle number plates from images, extracts the characters using Optical Character Recognition (OCR), and strictly validates them against the standard **Indian number plate format** (e.g., `MH 12 DE 1433`, `RJ 14 CV 0002`).

Instead of just running basic OCR, this project features a custom **Sliding Window Search Algorithm** and **Bi-Directional Auto-Correction** to fix common AI misreads (like confusing a `4` for an `L`, or a `0` for an `O`), making it highly accurate even on noisy images. 

Every valid detection is automatically logged with a timestamp into a `results.csv` database.

## ✨ Key Features
- **Modern Web Dashboard:** Built with Streamlit for a clean, user-friendly drag-and-drop interface.
- **Batch Processing:** Upload multiple images at once and watch the AI process them in seconds.
- **Noise Immunity:** Automatically ignores watermarks, screws, and the "IND" tag found on modern plates.
- **Smart Auto-Correction:** Uses positional logic to fix OCR mistakes. (e.g., If the AI reads `HR26DQ555I`, it knows the last character *must* be a digit and mathematically corrects the `I` to a `1`).

## ⚙️ How it works under the hood
1. **Resolution Scaling:** Images are dynamically scaled to 1000px wide. This optimizes the image so the OCR engine can read blurry or small letters with much higher accuracy.
2. **Text Extraction:** EasyOCR scans the entire frame to extract all visible text.
3. **Horizontal Merging:** The engine merges text boxes that are on the same horizontal plane (fixing plates where numbers are spaced far apart).
4. **The Sliding Window:** A 9-to-10 character "magnifying glass" slides across the messy OCR text to hunt for the actual plate, bypassing surrounding garbage text.
5. **Validation & Correction:** The candidate text is checked against the Indian plate regex (`SS DD LL DDDD`). The Bi-Directional dictionary forces numbers into digits and letters into alphabets based on their correct positions.
6. **Logging:** Validated plates are drawn with a green bounding box and saved to `results.csv`.

## 💻 Tech Stack
- **Python** (Core Logic)
- **Streamlit & Pandas** (Web Dashboard & Data Management)
- **OpenCV** (Image processing & bounding boxes)
- **EasyOCR** (Deep-learning text recognition)
- **Regex** (Format validation)

## 🚀 Setup Instructions (Windows)
Open Command Prompt or PowerShell in the project folder and run:

```powershell
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
venv\Scripts\activate

# 3. Install the required libraries
pip install -r requirements.txt