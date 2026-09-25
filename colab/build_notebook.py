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
import os, subprocess, urllib.error, urllib.request
from google.colab import userdata
os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN").strip()  # Colab Secrets → GH_TOKEN, bật "Notebook access"
try:  # Fail here, not inside the agent loop, when the token is wrong or lacks access to REPO.
    urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/repos/{{REPO}}",
        headers={{"Authorization": f"Bearer {{os.environ['GH_TOKEN']}}"}}), timeout=30)
except urllib.error.HTTPError as e:
    raise SystemExit(f"GH_TOKEN không dùng được với {{REPO}} (HTTP {{e.code}}): "
                     + ("token sai/hết hạn/đã xoá" if e.code == 401 else "token chưa được cấp quyền cho repo này"))
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


VIDEO_INTRO = f'''# Video → tour 3D kiểu Matterport trên Colab
Notebook độc lập: clone repo public [`{REPO}`](https://github.com/{REPO}/tree/{BRANCH}) (không cần token), chạy từng model một và hiện kết quả trung gian.
Chạy tuần tự từ trên xuống; mỗi bước lưu kết quả, nên chạy lại chỉ làm tiếp phần chưa xong.

## Model dùng trong pipeline
| Bước | Model / công cụ | Nguồn | Vai trò |
|---|---|---|---|
| 1 | FFmpeg + ORB/RANSAC (OpenCV) | `studio/sources.py`, `studio/selection.py` | Trích frame, bỏ ảnh mờ/trùng, kiểm tra các frame liền nhau có vùng chung |
| 2 | **VGGT-1B** (Meta) | [`facebook/VGGT-1B`](https://huggingface.co/facebook/VGGT-1B), repo `facebookresearch/vggt` | Một lần suy luận cho mọi frame: vị trí camera + depth → point cloud + COLMAP |
| 2b | VGGSfM tracker + **pycolmap bundle adjustment** (`demo_colmap.py --use_ba` của VGGT, môi trường Python 3.11 riêng) | repo `facebookresearch/vggt` | Tinh chỉnh pose camera (`POSE_BA`) |
| 3a | **3DGUT** (3DGRUT, NVIDIA), chiến lược MCMC | submodule `3DGRUT-ArtiFixer` của ArtiFixer | Train Gaussian Splat gốc từ ảnh thật (train riêng cho cảnh này, không có trọng số sẵn) |
| 3b | **MoGe-2** (Microsoft) | `microsoft/MoGe` | Ước lượng scale theo mét cho điều kiện camera của ArtiFixer |
| 3c | Text encoder **umt5** của Wan2.1 | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Mã hoá một caption cố định (thay Qwen3-VL-30B của ArtiFixer) |
| 3d | **ArtiFixer 1.3B** (NVIDIA, SIGGRAPH 2026) | [`nv-tlabs/ArtiFixer`](https://github.com/nv-tlabs/ArtiFixer) @ `a392c4d`, checkpoint [`nvidia/ArtiFixer`](https://huggingface.co/nvidia/ArtiFixer) trên nền Wan2.1-T2V-1.3B | Video diffusion sửa floater/lỗ trên các khung render từ một quỹ đạo camera mới |
| 3e | **ArtiFixer3D** | cùng repo ArtiFixer | Distill ảnh thật + khung đã sửa thành splat mới (`splat.ply`) |
| 4 | Cắt gọn + viewer tham quan | `colab/pipeline.py`, `colab/viewer.html` (Three.js + Spark) | Bỏ Gaussian ngoài vùng quay, floater; tour bấm-để-đi kiểu Matterport |

**Có dùng [ArtiFixer](https://github.com/nv-tlabs/ArtiFixer):** có, bước 3 (prep 3DGUT, inference 1.3B, ArtiFixer3D).
**Không dùng:** thư viện **nerfstudio** (chỉ ghi file `transforms.json` theo định dạng của nó, không import nerfstudio), COLMAP SfM đầy đủ (pose lấy từ VGGT, chỉ dùng bundle adjustment của pycolmap), OpenSplat (chỉ app Mac), Qwen3-VL.

**Viewer:** khi đứng tại một điểm, viewer hiện **ảnh gốc** của frame đó (như Matterport hiện ảnh 360°); splat dùng cho phần ngoài khung ảnh, lúc di chuyển, Dollhouse (cắt trần) và Mặt bằng.

## Yêu cầu và thời gian
- Runtime **GPU A100 80 GB hoặc H100** (+ High-RAM). A100 40 GB: bật `LOW_MEMORY`.
- `setup` lần đầu ~7–30 phút (tải ~30 GB model, cache vào Drive). Thử nghiệm courtyard 8 ảnh trên A100: VGGT 37 giây, 3DGUT + ArtiFixer ~11 phút, ArtiFixer3D 30.000 bước ~2 giờ 20 phút.
- Video: quay chậm, liên tục, vùng chung giữa các khung ≥60%. Chỉ vùng đã quay mới được dựng.'''

VIDEO_PARAMS = f'''#@title Tham số
VIDEO = "/content/drive/MyDrive/3D/IMG_6608.MOV"  #@param {{type:"string"}}
#@markdown Đường dẫn video trên Google Drive (sau khi mount) hoặc URL `https://…`.
NAME = "img6608"  #@param {{type:"string"}}
TARGET_FRAMES = 32  #@param {{type:"integer"}}
RECONSTRUCTION_STEPS = 10000  #@param {{type:"integer"}}
ARTIFIXER3D_STEPS = 30000  #@param {{type:"integer"}}
#@markdown Giảm `ARTIFIXER3D_STEPS` (vd 15000) để nhanh hơn ~1 giờ, đổi lại chất lượng thấp hơn.
TRAJECTORY_FRAMES = 81  #@param {{type:"integer"}}
LOW_MEMORY = False  #@param {{type:"boolean"}}
POSE_BA = True  #@param {{type:"boolean"}}
#@markdown `POSE_BA`: tinh chỉnh pose camera bằng bundle adjustment chính thức của VGGT (thêm ~5 phút, ảnh nét hơn khi pose lệch).
REPO = "{REPO}"  #@param {{type:"string"}}
BRANCH = "{BRANCH}"  #@param {{type:"string"}}'''

VIDEO_SETUP = '''#@title 0 · Clone code, mount Drive, cài đặt model
import json, os, shlex, subprocess, sys
from pathlib import Path
from IPython.display import Image, display
if not Path("/content/drive/MyDrive").is_dir():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:  # papermill/không có Drive: vẫn chạy được với VIDEO là URL hoặc file cục bộ
        print("Không mount Drive:", e)
CODE = Path("/content/3D-ify-everything")
url = f"https://github.com/{REPO}.git"
if not (CODE/".git").exists():
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, url, str(CODE)], check=True)
else:
    subprocess.run(["git", "-C", str(CODE), "fetch", "-q", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(CODE), "reset", "-q", "--hard", f"origin/{BRANCH}"], check=True)
os.chdir(CODE)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)

def pipeline(*args):
    """Chạy `python -m colab.pipeline …`, in log trực tiếp, báo lỗi (dừng notebook) nếu thất bại."""
    cmd = [sys.executable, "-m", "colab.pipeline", *map(str, args)]
    print("$", shlex.join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout: print(line, end="", flush=True)
    if p.wait(): raise RuntimeError(f"pipeline {args[0]} thất bại (exit {p.returncode}); xem log ở trên")

def grid(paths, cols=8, width=1600, labels=None):
    """Ghép ảnh thành một lưới JPEG nhỏ để hiện trong notebook."""
    from PIL import Image as PILImage, ImageDraw
    paths = [Path(p) for p in paths]
    if not paths: print("(không có ảnh)"); return
    with PILImage.open(paths[0]) as first: aspect = first.height / first.width
    cols = min(cols, len(paths))
    cell = width // cols; h = int(cell * aspect)  # cells follow the frame shape (portrait phone video stays tight)
    ims = []
    for p in paths:
        with PILImage.open(p) as im:
            im = im.convert("RGB"); im.thumbnail((cell, h)); ims.append(im)
    sheet = PILImage.new("RGB", (cell * cols, h * ((len(ims) + cols - 1) // cols)), (16, 19, 18))
    for k, im in enumerate(ims):
        x, y = (k % cols) * cell, (k // cols) * h
        sheet.paste(im, (x, y))
        if labels: ImageDraw.Draw(sheet).text((x + 4, y + 4), str(labels[k]), fill=(255, 255, 255))
    out = Path("/tmp")/f"grid-{abs(hash(tuple(map(str, paths))))}.jpg"
    sheet.save(out, quality=80); display(Image(str(out)))

BASE = Path("/content/work")/NAME
RUN_ARGS = ["--source", "video", "--input", VIDEO, "--name", NAME, "--target-frames", TARGET_FRAMES,
            "--reconstruction-steps", RECONSTRUCTION_STEPS, "--artifixer3d-steps", ARTIFIXER3D_STEPS,
            "--trajectory-frames", TRAJECTORY_FRAMES] + (["--low-memory"] if LOW_MEMORY else []) + (["--pose-ba"] if POSE_BA else [])
pipeline("check")
pipeline("setup")'''

VIDEO_STEP1 = '''#@title 1 · Video → frame (FFmpeg, chọn frame rõ + có vùng chung)
pipeline("run", *RUN_ARGS, "--stages", "inputs")
video = json.loads((BASE/"project"/"video.json").read_text())
sel = json.loads((BASE/"run"/"selection.json").read_text())
print(f"Video {video['duration_seconds']:.1f} giây · {video['candidate_count']} frame ứng viên · "
      f"chọn {sel['selected_count']} · loại {sel['rejected_count']} ảnh mờ/gần trùng · liên kết đủ: {sel['connected']}")
grid([BASE/"project"/m["thumbnail"] for m in sel["images"]],
     labels=[f"{m.get('timestamp_seconds', 0):.1f}s" for m in sel["images"]])'''

VIDEO_STEP2 = '''#@title 2 · VGGT-1B: vị trí camera + depth trong một lần suy luận
pipeline("run", *RUN_ARGS, "--stages", "vggt")
geo = json.loads((BASE/"run"/"geometry"/"metrics.json").read_text())
ba_path = BASE/"run"/"geometry"/"pose-ba.json"
if ba_path.exists():
    ba = json.loads(ba_path.read_text())
    print("Bundle adjustment:", f"{ba['registered']}/{ba['images']} ảnh, chỉnh hướng camera trung bình {ba['mean_rotation_change_deg']:.2f}° (tối đa {ba['max_rotation_change_deg']:.2f}°)"
          if ba.get("applied") else f"không áp dụng ({ba.get('reason')})")
print(f"{geo['camera_count']} camera · {geo['point_count']:,} điểm · suy luận {geo['inference_seconds']:.1f} giây · "
      f"VRAM đỉnh {geo['peak_reserved_bytes'] / 2**30:.1f} GB · "
      f"điểm nhất quán giữa các góc nhìn {sum(geo['cross_view_supported_fraction']) / len(geo['cross_view_supported_fraction']):.0%}")
print("Depth tương đối (đỏ = xa, xanh = gần; đen = bị loại vì confidence thấp):")
grid(sorted((BASE/"run"/"geometry").glob("depth_*.png")))'''

VIDEO_STEP3 = '''#@title 3 · 3DGUT → ArtiFixer 1.3B → ArtiFixer3D (bước dài nhất)
#@markdown 3a. Dựng scene COLMAP với một calibration chung, tạo quỹ đạo camera mới đi qua các góc đã quay.
#@markdown 3b. Train **3DGUT** MCMC (`RECONSTRUCTION_STEPS`), render quỹ đạo; **MoGe-2** ước lượng scale.
#@markdown 3c. **ArtiFixer 1.3B** sửa các khung render; 3d. **ArtiFixer3D** distill thành splat mới.
pipeline("run", *RUN_ARGS, "--stages", "artifixer")
af = json.loads((BASE/"stages"/"artifixer.json").read_text())
pred = Path(af["pred"]); rendered = pred.parent/"rendered"
frames = sorted(pred.glob("*.png"))
pick = [frames[i] for i in range(0, len(frames), max(1, len(frames) // 4))][:4]
print("Hàng trên: render 3DGUT gốc · hàng dưới: ArtiFixer đã sửa")
grid([rendered/p.name for p in pick] + pick, cols=4)'''

VIDEO_STEP4 = '''#@title 4 · Đóng gói: cắt gọn splat, viewer, results.zip
pipeline("run", *RUN_ARGS, "--stages", "package")
m = json.loads((BASE/"enhance"/"metrics.json").read_text())
for name, t in (m.get("tidy") or {}).items():
    print(f"{name}.ply: {t['before']:,} → {t['after']:,} Gaussian (ngoài vùng {t['outside_box']:,}, trong suốt {t['transparent']:,}, "
          f"quá lớn {t['oversized']:,}, cô lập {t['isolated']:,})")
print("Scale metric:", m.get("metric_scale"))
if (BASE/"enhance"/"compare.jpg").exists(): display(Image(str(BASE/"enhance"/"compare.jpg"), width=900))
print("\\n".join(m["files"][:20]))'''

VIDEO_STEP5 = '''#@title 5 · Tour 3D trong trình duyệt (kéo để nhìn, bấm vòng tròn trên sàn để đi)
PORT = 8765
subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--directory", str(BASE/"enhance")],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    from google.colab import output
    output.serve_kernel_port_as_window(PORT, path="/index.html")
except Exception as e:
    print("Mở viewer trong Colab không được ở môi trường này:", e)
    print(f"Tải results.zip, giải nén, chạy `python -m http.server {PORT}` trong thư mục đó rồi mở http://127.0.0.1:{PORT}")'''

VIDEO_STEP6 = '''#@title 6 · Lưu kết quả
zip_path = BASE/"results.zip"
print(f"{zip_path} · {zip_path.stat().st_size / 2**20:.0f} MiB (đã tự sao lưu vào Drive/courtyard-results nếu Drive được mount)")
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as e:
    print("Không tải trực tiếp được ở môi trường này:", e)'''


def video_tour():
    return notebook([
        new_markdown_cell(VIDEO_INTRO),
        new_code_cell(VIDEO_PARAMS, metadata={'tags': ['parameters']}),
        new_code_cell(VIDEO_SETUP),
        new_markdown_cell('## 1 · Video → frame\nFFmpeg trích tối đa 1.200 frame trải đều toàn video. Bộ chọn lấy `TARGET_FRAMES` frame rõ nhất theo từng khoảng thời gian, bỏ ảnh mờ và gần trùng, rồi kiểm tra ORB/RANSAC để các frame liền nhau có vùng chung. Thiếu liên kết thì thêm frame trung gian (tới 48 rồi 64) hoặc báo đoạn cần quay lại.'),
        new_code_cell(VIDEO_STEP1),
        new_markdown_cell('## 2 · VGGT-1B\nMọi frame đã chọn vào **một** lần suy luận (CUDA, bf16 cho aggregator). VGGT trả về vị trí/hướng camera, intrinsics và depth; điểm được giữ khi có confidence cao và khớp depth với góc nhìn lân cận. Kết quả ghi thành `points.ply`, `cameras.json` và COLMAP `sparse/0`, làm đầu vào cho 3DGUT (không chạy COLMAP SfM).'),
        new_code_cell(VIDEO_STEP2),
        new_markdown_cell('## 3 · 3DGUT → ArtiFixer → ArtiFixer3D\nĐây là phần dùng repo [nv-tlabs/ArtiFixer](https://github.com/nv-tlabs/ArtiFixer). 3DGUT dựng splat gốc từ ảnh thật; splat này còn floater và lỗ ở góc chưa quay kỹ. ArtiFixer (video diffusion 1.3B) nhìn các khung render từ một đường đi camera mới và vẽ lại cho sạch; ArtiFixer3D train lại splat từ ảnh thật cộng các khung đã sửa. AI có thể "vẽ thêm" chi tiết không có thật ở vùng thiếu dữ liệu.'),
        new_code_cell(VIDEO_STEP3),
        new_markdown_cell('## 4 · Đóng gói\nChuyển PLY của 3DGRUT sang định dạng 3DGS gọn, cắt bỏ Gaussian ngoài vùng đã quay / gần trong suốt / quá lớn / cô lập, và đóng gói viewer tour cùng `results.zip`.'),
        new_code_cell(VIDEO_STEP4),
        new_code_cell(VIDEO_STEP5),
        new_code_cell(VIDEO_STEP6),
    ], gpu='A100')


def main():
    for name, nb in (('agent_bootstrap.ipynb', agent_bootstrap()), ('courtyard_artifixer.ipynb', full_pipeline()),
                     ('video_tour.ipynb', video_tour())):
        nbformat.validate(nb)
        nbformat.write(nb, HERE/name)
        print(HERE/name)


if __name__ == '__main__':
    main()
