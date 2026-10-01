# A Real Options-Based Valuation Framework for Renewable Energy Companies

A code repository for a Master's Thesis in Financial Engineering.

<img src="images/Norwegian University of Science and Technology - farger - bredde.png" align="center" height="50">

## File Structure

```
/NotlandTaraldsen
│
├── /src/                # Source code for the project
├── /data/               # Data files
├── /documents/          # Some documents
├── /results/            # Sensitivity results and model results
├── requirements.txt     # Python dependencies
├── README.md            # Project overview (this file)
└── .gitignore           # Git ignore rules
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- `pip` (Python package manager)

### Setting Up a Virtual Environment

1. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

2. Activate the virtual environment:
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project
- Ensure the virtual environment is activated.
- Run the desired scripts from the `/src/` directory.
- You can only run results.py after running at least one NT_2025 model file.


## Additional Notes
- Update `requirements.txt` after installing new packages:
  ```bash
  pip freeze > requirements.txt
  ```
- Deactivate the virtual environment when done:
  ```bash
  deactivate
  ```

---
> Software developed by:\
> **Per Inge Notland**, M.Sc student at NTNU | perino@stud.ntnu.no \
> **David Taraldsen**, M.Sc student at NTNU | davideta@stud.ntnu.no \
> Date: 05.06.2025