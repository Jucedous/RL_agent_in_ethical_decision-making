# Project Setup

## Installation

Follow these steps to set up and run the project.
Trained reward model download link: https://drive.google.com/file/d/1JP-Vsi1Wyr3r790ZroileXyankeGP1Np/view?usp=sharing

### Steps

1. **Create and activate a virtual environment using Python 3.10**:

   ```sh
   python3.10 -m venv venv
   venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Run the FastAPI server**:

   ```sh
   uvicorn main:app --reload
   ```

4. **Access the API in a browser**: Open your browser and navigate to:

   ```
   http://127.0.0.1:8000/
   ```

---


