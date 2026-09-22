import json
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from studio.api import app
(root/'openapi.json').write_text(json.dumps(app.openapi(),indent=2))
