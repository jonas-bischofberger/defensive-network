## Folder structure

- defensive_network/: 
  - assets/: All files that are not code, e.g. xT weights
  - models/: All analytics models such as expected threat, expected receiver, involvement, tracking-event synchronization, formations, ...
  - parse/: To access data.
  - tests/: Tests to ensure reliability of the code (use pytest to run)
  - utility/: Various helper functions.
- old/: Old files we don't need anymore (but keep them just in case)
- scripts/: Various Streamlit dashboards and Python scripts to analyse the data
- secrets/: Token etc. for accessing Google Drive


## Installation

1. Clone the repository:

```
git clone https://github.com/jonas-bischofberger/defensive-network
cd defensive-network
```

2. Create and activate a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Run some analysis script!

```
streamlit run scripts/explore/Explore_defensive_network.py
streamlit run scripts/xt_statsbomb_correlation.py
streamlit run scripts/responsibility_and_google_drive.py
```
