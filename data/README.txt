
IFMBE Scientific Challenge 2025
--------------------------------

Welcome to the IFMBE Scientific Challenge at IUPESM 2025!

You have received:
- `trainData.csv`: Training dataset
- `valData.csv`: Validation dataset (for local testing)
- `example_script.py`: Example Python script to guide your submission

---

🧠 Your Task:
Develop a Python script (e.g., `myProgram.py`) that:
1. Trains your model using `trainData.csv`
2. Loads the test data (which will be provided by organizers)
3. Predicts:
   - Length of Stay (integer)
   - Discharge Type (0 = survival, 1 = death)
4. Outputs two CSV files:
   - `LSestimation.csv`: vector of predicted length of stay
   - `DTestimation.csv`: vector of predicted discharge type

Your script **must be executable** and **contain the trained model or retrain it** when executed. It will be run in a controlled environment using only the test data features.

Do not use or request access to any external files or internet connections in your code.

---

📤 Submission Instructions:
Submit a ZIP file containing:
- `myProgram.py`: Your main Python script
- `model files`: If needed, include any saved model weights (e.g., `.pkl`, `.pt`, `.joblib`)
- `README.txt`: (Optional) Description of your approach and dependencies
- Do NOT include `valData.csv` or `trainData.csv` in your submission

Your script will be tested on an unseen `testData.csv` with the same format.

---

📍 Submission Portal:
https://sc-iupesm-2025.dei.uc.pt/

📅 Deadline:
[Check the official website for updates]

Good luck!
