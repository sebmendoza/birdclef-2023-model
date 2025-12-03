## Generate requirements.txt

With your virtual environment activated and when you add new packages, run: `pip freeze > requirements.txt`

## Build from requirements.txt

```
# Create and activate a fresh virtual environment
python -m venv venv
source venv/bin/activate

# Install from requirements.txt
pip install -r requirements.txt
```
