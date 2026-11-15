# Streamlit Cloud entry point
# Redirects to the pressure dashboard
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Run the dashboard
exec(open(Path(__file__).parent / "src" / "superr_selector" / "pressure" / "dashboard.py").read())
