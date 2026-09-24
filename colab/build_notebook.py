"""Generate the Colab notebooks from code so they never drift from colab/*.py.

    python colab/build_notebook.py
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
REPO = 'nguyennguyenphuc/3D-ify-everything'
BRANCH = 'feature/courtyard-studio'

CLONE = f'''#@title 1 · Clone code + mount Google Drive (cache model ~30 GB, lưu kết quả)
REPO = "{REPO}"  #@param {{type:"string"}}
BRANCH = "{BRANCH}"  #@param {{type:"string"}}
MOUNT_DRIVE = True  #@param {{type:"boolean"}}
import os, subprocess
from google.colab import userdata
os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN")  # Colab Secrets → GH_TOKEN, bật "Notebook access"
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
WORK = "/content/3D-ify-everything"
url = f"https://x-access-token:{{os.environ['GH_TOKEN']}}@github.com/{{REPO}}.git"
if not os.path.exists(WORK):
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, url, WORK], check=True)
else:
    subprocess.run(["git", "-C", WORK, "fetch", "-q", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", WORK, "reset", "-q", "--hard", f"origin/{{BRANCH}}"], check=True)
os.chdir(WORK)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)'''

TOKEN_HELP = f'''### Chuẩn bị một lần
1. **Runtime → Change runtime type → GPU: H100** (hoặc **A100** + bật **High-RAM**). ArtiFixer 1.3B cần ~40–80 GB VRAM.
2. Tạo GitHub token *fine-grained* chỉ cho repo `{REPO}`: **Contents: Read and write**, **Issues: Read and write**, **Metadata: Read**.
3. Colab → biểu tượng 🔑 **Secrets** → thêm `GH_TOKEN` = token trên, bật **Notebook access**.

Token chỉ nằm trong Colab Secrets; agent không đưa token vào môi trường của job.'''


def notebook(cells, gpu='A100'):
    nb = new_notebook(cells=cells)
    nb.metadata.update({'accelerator': 'GPU', 'colab': {'gpuType': gpu, 'machine_shape': 'hm', 'provenance': []},
                        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}})
    return nb


def agent_bootstrap():
    return notebook([
        new_markdown_cell(f'''# Kết nối Colab GPU ↔ GitHub (agent)
Notebook này biến runtime Colab thành máy GPU nhận lệnh qua GitHub. Sau khi chạy 2 ô bên dưới, bất kỳ máy nào truy cập được `api.github.com` (kể cả Claude Code trên cloud) có thể gửi lệnh, xem log và lấy kết quả:

```bash
python -m colab.remote status
python -m colab.remote run "python -m colab.pipeline check"
python -m colab.remote run "python -m colab.pipeline setup" --timeout 90
python -m colab.remote run "python -m colab.pipeline run --source courtyard" --timeout 240
```

Kênh điều khiển là issue **`[colab-agent] control channel`** trong repo. Log từng job nằm ở comment trạng thái, file nhỏ ở nhánh `colab-runs/<id>`, file lớn (splat `.ply`, `results.zip`) ở release `colab-run-<id>`. Chỉ comment của owner/collaborator được thực thi.

{TOKEN_HELP}'''),
        new_code_cell(CLONE),
        new_code_cell('''#@title 2 · Chạy agent (để ô này chạy; bấm ■ để dừng)
#@markdown Thêm login GitHub khác được phép gửi lệnh (cách nhau dấu phẩy), nếu cần:
ALLOW = ""  #@param {type:"string"}
!python -m colab.agent --repo {REPO} --workdir {WORK} --branch {BRANCH} --allow "{ALLOW}"'''),
    ], gpu='H100')


def full_pipeline():
    return notebook([
        new_markdown_cell(f'''# Courtyard Studio trên Colab: VGGT → 3DGUT → ArtiFixer → ArtiFixer3D
Chạy toàn bộ pipeline trên GPU Colab, không cần máy Mac:

1. **VGGT-1B** (CUDA, bf16) ước lượng pose + depth từ ảnh/video → COLMAP.
2. **3DGUT MCMC** dựng Gaussian Splat gốc; render một quỹ đạo camera mới đi qua các góc đã quay.
3. **ArtiFixer 1.3B** (video diffusion) sửa floater/lỗ và điền vùng thiếu trên quỹ đạo đó.
4. **ArtiFixer3D** distill ảnh thật + ảnh đã sửa thành splat mới → `splat.ply`.

Lần đầu `setup` mất ~20–30 phút (tải ~30 GB model, cache vào Drive). Scene 8 ảnh ETH3D chạy ~45–75 phút trên H100.

{TOKEN_HELP}'''),
        new_code_cell(CLONE),
        new_code_cell('''#@title 2 · Kiểm tra GPU / môi trường
!python -m colab.pipeline check'''),
        new_code_cell('''#@title 3 · Cài đặt (torch 2.11 cu128, 3DGRUT, ArtiFixer, tải model) — chạy lại mỗi phiên, model lấy từ cache
!python -m colab.pipeline setup'''),
        new_code_cell('''#@title 4 · Chạy pipeline
SOURCE = "courtyard"  #@param ["courtyard", "video", "images"]
INPUT = ""  #@param {type:"string"}
#@markdown `INPUT`: đường dẫn video/thư mục ảnh trên Drive (vd `/content/drive/MyDrive/phong.mp4`) hoặc URL. Bỏ trống với courtyard.
NAME = "courtyard"  #@param {type:"string"}
TARGET_FRAMES = 32  #@param {type:"integer"}
TRAIN_IMAGE_MAX = 1536  #@param {type:"integer"}
RECONSTRUCTION_STEPS = 10000  #@param {type:"integer"}
ARTIFIXER3D_STEPS = 30000  #@param {type:"integer"}
TRAJECTORY_FRAMES = 81  #@param {type:"integer"}
LOW_MEMORY = False  #@param {type:"boolean"}
#@markdown Bật `LOW_MEMORY` trên A100 40 GB.
import shlex
cmd = ["python", "-m", "colab.pipeline", "run", "--source", SOURCE, "--name", NAME,
       "--target-frames", str(TARGET_FRAMES), "--train-image-max", str(TRAIN_IMAGE_MAX),
       "--reconstruction-steps", str(RECONSTRUCTION_STEPS), "--artifixer3d-steps", str(ARTIFIXER3D_STEPS),
       "--trajectory-frames", str(TRAJECTORY_FRAMES)]
if INPUT: cmd += ["--input", INPUT]
if LOW_MEMORY: cmd += ["--low-memory"]
print(shlex.join(cmd))
!{shlex.join(cmd)}'''),
        new_code_cell('''#@title 5 · Xem so sánh + thông số
import json
from IPython.display import Image, Video, display
out = f"/content/work/{NAME}/enhance"
m = json.load(open(f"{out}/metrics.json"))
print(json.dumps({k: m.get(k) for k in ("name", "gpu", "artifixer3d_ply", "baseline_ply", "metric_scale", "stages", "note")}, indent=2, ensure_ascii=False))
import os
if os.path.exists(f"{out}/compare.jpg"): display(Image(f"{out}/compare.jpg"))
if os.path.exists(f"{out}/preview.mp4"): display(Video(f"{out}/preview.mp4", embed=True, width=720))'''),
        new_code_cell('''#@title 6 · Viewer 3D trong trình duyệt (splat ArtiFixer3D / splat gốc / point cloud)
import subprocess, sys
from google.colab import output
PORT = 8765
subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--directory", f"/content/work/{NAME}/enhance"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
output.serve_kernel_port_as_window(PORT, path="/index.html")'''),
        new_code_cell('''#@title 7 · Tải results.zip (đã tự lưu vào Drive/courtyard-results nếu mount Drive)
from google.colab import files
files.download(f"/content/work/{NAME}/results.zip")'''),
    ])


def main():
    for name, nb in (('agent_bootstrap.ipynb', agent_bootstrap()), ('courtyard_artifixer.ipynb', full_pipeline())):
        nbformat.validate(nb)
        nbformat.write(nb, HERE/name)
        print(HERE/name)


if __name__ == '__main__':
    main()
