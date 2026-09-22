import json
import subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for name,rev in json.loads((root/'upstream.lock.json').read_text()).items():
    path = root/'vendor'/name
    url = {'vggt':'https://github.com/facebookresearch/vggt.git','OpenSplat':'https://github.com/WebODM/OpenSplat.git'}[name]
    if not path.exists(): subprocess.run(['git','clone',url,str(path)],check=True)
    current = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if current != rev:
        subprocess.run(['git','-C',str(path),'fetch','origin',rev],check=True)
        subprocess.run(['git','-C',str(path),'checkout',rev],check=True)
