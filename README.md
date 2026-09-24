# Courtyard Studio — VGGT MPS → OpenSplat Metal

Web UI local để tái dựng từ video/ảnh, xem geometry và train Gaussian Splatting trên Apple Silicon. Server chỉ bind `127.0.0.1`. Mỗi project giữ nguồn ảnh riêng; mỗi run giữ cấu hình, log, event, report và artifact riêng.

## Điều kiện máy

1. Cài Xcode từ App Store, mở Xcode một lần và chấp nhận license.
2. Chọn Xcode đầy đủ rồi tải Metal Toolchain:

```zsh
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
xcrun -sdk macosx metal --version
```

Lệnh cuối phải in phiên bản compiler. Studio từ chối OpenSplat CPU fallback; `PYTORCH_ENABLE_MPS_FALLBACK=0` luôn được đặt trong worker.

## Cài đặt và khởi động

```zsh
cd /Users/nguyennp/AI/3D
scripts/setup.sh
scripts/start.sh
```

Mở `http://127.0.0.1:8000`. `setup.sh` khóa source upstream tại [upstream.lock.json](upstream.lock.json), tải VGGT-1B đúng revision và chỉ xác nhận OpenSplat khi compile command có `USE_MPS`, binary có `metallib` và fingerprint hợp lệ.

Trên macOS, `setup.sh` liên kết runtime OpenMP của LibTorch với runtime Homebrew mà OpenCV/OpenBLAS dùng. Bản LibTorch gốc được giữ thành `libomp.torch-original.dylib` trong `.venv` để có thể khôi phục.

Có thể kiểm tra riêng:

```zsh
.venv/bin/python -m studio.environment
.venv/bin/python -m pytest
npm run build --prefix web
```

## Chạy courtyard

1. Tạo project và chọn **Tải ETH3D courtyard**. Archive chính thức được lưu ở `data/downloads/`; calibration ETH3D chỉ chọn 8 ảnh overlap và tạo `courtyard-selection.json`.
2. Đây là fixture kiểm thử riêng: chọn đúng 8 ảnh rồi chạy **VGGT MPS → Geometry**. Worker chạy 8 ảnh trong một inference FP32 MPS và giải phóng model trước training.
3. Kiểm tra `geometry/metrics.json`, `depth_*.png`, `points.ply`, `cameras.json` và `sparse/0/*.bin` trong artifact.
4. Chạy **Train OpenSplat Metal**. Worker chạy smoke 100 bước rồi tối đa 10.000 bước, snapshot mỗi 50 bước. Khi hết thời gian, snapshot đã ghi vẫn được giữ.

Depth được ghi là thang đo tương đối. Camera sử dụng OpenCV `world-to-camera`; `cameras.json` ghi rõ ma trận biến đổi viewer. Point cloud loại padding, depth âm/không hữu hạn, confidence thấp và điểm không nhất quán với góc nhìn lân cận. Mức 100.000 điểm chỉ bật sau smoke Metal thành công. Setup ghép LibTorch và Homebrew OpenCV/OpenBLAS vào cùng `libomp` để tránh nạp hai OpenMP runtime trên macOS; log vẫn bắt buộc phải chứa `Using MPS`.

## Dựng phòng từ video

Chọn **Upload và dựng phòng 3D** (mặc định bật) rồi gửi video điện thoại quay liên tục quanh phòng. Sau khi upload xong, job có ngân sách tối đa 30 phút:

1. FFprobe đọc duration; FFmpeg lấy tối đa 1.200 frame xuyên toàn bộ video và lưu PTS nguồn cho mỗi frame.
2. Bộ chọn mặc định lấy 32 frame trải đều, ưu tiên ảnh rõ, bỏ ảnh gần trùng, kiểm tra ORB/RANSAC overlap; thiếu liên kết sẽ thêm ảnh trung gian tới 48 rồi 64 hoặc báo khoảng thời gian cần quay bổ sung.
3. VGGT chạy mọi frame đã chọn trong một inference MPS chung. Số frame không tự giảm khi thiếu bộ nhớ; UI chỉ cho chọn mức đã đo thành công trên máy này.
4. OpenSplat chạy smoke 100 bước rồi tối đa 10.000 bước trong thời gian còn lại. `selection.json`, `mps-memory.json`, `report.json`, depth, cameras, point cloud, snapshot và PLY cuối nằm trong artifact của run.

Người dùng có thể chuyển sang chọn thủ công để thêm/bỏ ảnh trên timeline trước khi chạy lại. Viewer mở Splat khi có kết quả; chọn camera theo timestamp, hoặc Point cloud/Depth để đối chiếu. Chỉ các bề mặt đã quay đủ góc mới có thể tái dựng; vùng bị che và mặt sau đồ vật không được suy diễn.

## Chạy toàn bộ trên Google Colab (không cần Mac)

Thư mục `colab/` chạy cùng pipeline trên GPU NVIDIA của Colab và thêm **ArtiFixer** (NVIDIA, SIGGRAPH 2026) để sửa floater/lỗ và điền vùng thiếu:

1. **VGGT-1B CUDA bf16** (`studio/vggt.py`, dùng chung `studio/geometry.py`) → COLMAP `run/sparse/0`, ảnh train 1536 px, tới 128 frame.
2. **3DGUT MCMC** (3DGRUT) dựng splat gốc, render một quỹ đạo camera mới đi qua các góc đã quay.
3. **ArtiFixer 1.3B** sửa các khung hình của quỹ đạo đó; **ArtiFixer3D** distill ảnh thật + ảnh đã sửa thành `splat.ply`.
4. Đóng gói `enhance/` gồm `splat.ply`, `baseline.ply`, `points.ply`, `compare.jpg`, `preview.mp4`, `metrics.json` và `index.html` (viewer Spark). Có thêm `results.zip`, bản sao lưu trên Drive nếu đã mount.

ArtiFixer không tăng độ phân giải ảnh nguồn. Để chi tiết gần kiểu Matterport, hãy quay chậm, chồng lấn ≥60% và đi 2 vòng ở 2 độ cao.

### Cách 1 · Chạy tay trong Colab

Mở `colab/courtyard_artifixer.ipynb` ([Open in Colab](https://colab.research.google.com/github/Thanhjash/3D-ify-everything/blob/feature/courtyard-studio/colab/courtyard_artifixer.ipynb)), chọn GPU H100 hoặc A100 + High-RAM rồi chạy lần lượt các ô.

### Cách 2 · Điều khiển Colab từ máy khác qua GitHub

Máy điều khiển (kể cả Claude Code trên cloud) chỉ cần truy cập `api.github.com`; không cần tunnel và không tải gì về máy cá nhân.

1. Tạo GitHub token *fine-grained* chỉ cho repo này với quyền Contents RW, Issues RW, Metadata R. Lưu token vào Colab Secrets dưới tên `GH_TOKEN`.
2. Mở `colab/agent_bootstrap.ipynb` trên Colab (GPU H100/A100), chạy ô 1 rồi ô 2. Để ô 2 chạy liên tục.
3. Từ máy điều khiển:

```bash
python -m colab.remote status                                   # heartbeat + GPU
python -m colab.remote run "python -m colab.pipeline check"
python -m colab.remote run "python -m colab.pipeline setup" --timeout 90
python -m colab.remote run "python -m colab.pipeline run --source courtyard" --timeout 240 --name courtyard
python -m colab.remote files <job-id>
python -m colab.remote fetch <job-id> splat.ply out/splat.ply
```

Cơ chế: issue `[colab-agent] control channel` là hộp thư. Issue body chứa heartbeat. Mỗi comment job là một lệnh shell. Agent cập nhật comment trạng thái (log tail) mỗi 20 giây. File nhỏ trong `$COLAB_PUBLISH_DIR` được đẩy lên nhánh `colab-runs/<id>`, file lớn trong `$COLAB_RELEASE_DIR` lên release `colab-run-<id>`. Chỉ comment của owner/member/collaborator hoặc login trong `--allow` mới được chạy. Ai ghi được comment như vậy là chạy được lệnh trên runtime Colab, nên hãy giữ repo private và token chỉ cấp cho repo này. Trước mỗi job, agent `git reset --hard` về nhánh code mới nhất, nên sửa code chỉ cần push.

`python -m colab.pipeline selftest` chạy toàn bộ phần không cần model (scene tổng hợp → quỹ đạo → scene ArtiFixer → chuyển PLY → đóng gói) trên CPU. Test: `pytest tests/test_colab_*.py tests/test_vggt_device.py`.

## Nhập dữ liệu

- **Thư mục ảnh:** phải nằm dưới `STUDIO_IMPORT_ROOTS` (mặc định là workspace). Đặt ví dụ: `export STUDIO_IMPORT_ROOTS=/Users/nguyennp/Pictures:/Users/nguyennp/AI/3D` trước khi khởi động.
- **Video:** `POST /api/projects/{project_id}/video` với multipart `file`, `fps=1`, `reconstruct=true` và `target_frames=32`; FFmpeg trích tối đa 1.200 frame 1.920 px, tự áp rotation metadata, rồi tạo thumbnail. UI tự tạo project `Video · <tên file>` nếu project đang chọn đã có ảnh, nên không ghi đè nguồn hay run cũ.

API OpenAPI nằm trong `openapi.json`; client TypeScript sinh tại `web/src/generated.ts`. Mỗi job có SSE ở `/api/jobs/{id}/events` và hủy qua `POST /api/jobs/{id}/cancel`.

## Trạng thái nghiệm thu hiện tại

- MPS đã đo thật trên 16/32/48/64 ảnh một inference; mức 64 đạt với đỉnh MPS driver 20.5 GB.
- OpenSplat Metal đã smoke 100.000 điểm khởi tạo, 100 bước và PLY hợp lệ.
- Unit test kiểm tra depth unprojection/reprojection, resize intrinsics, đổi hệ camera, video dài 611 giây, frame gần cuối video, ảnh mờ/trùng, scene cut và API capacity gate.
- ETH3D courtyard đã tải và manifest 8 ảnh overlap đã tạo.
- Smoke `1516638fd53d4870b9202062167db7f0` dùng IMG_6608 (28.143 giây), chọn 29 frame, chạy VGGT chung và 10.000 bước Metal trong 381 giây. Splat 135.069 Gaussian hợp lệ, nhưng đây không phải nghiệm thu toàn phòng vì IMG_6608/IMG_6609 là cùng một clip ngắn byte-identical.

## Xem kết quả 3D

Mở `http://127.0.0.1:8000`, chọn project và run đã hoàn tất. Viewer tự chọn **Splat 3D** nếu có `training/splat.ply`; kéo chuột để xoay và cuộn để phóng to. Chọn **Camera 1…8** để trở về góc chụp, hoặc **Toàn cảnh** để xem bao quát. **Point cloud** là hình học thưa, có thanh chỉnh cỡ điểm; **Depth** đối chiếu độ sâu tương đối với ảnh nguồn. Camera chỉ vẽ khi bật **Hiện camera**, tránh che cảnh. Link dưới viewer tải file splat PLY.

Đã sửa ma trận camera row-major sang Three.js column-major, gắn SparkRenderer vào scene và chỉ render một lần mỗi frame. Kết quả tự tải lại khi job kết thúc; SSE đóng sau khi hoàn tất. Viewer ngừng render khi tab ẩn hoặc project đang xử lý GPU.

Kiểm tra trình duyệt với Chrome đã cài và server đang chạy:

```bash
cd web
rtk proxy node verify-viewer.mjs
```

Script dùng project courtyard có sẵn, kiểm tra các chế độ, tải lại trang và mô phỏng sự kiện hoàn tất bằng artifact thật (không chạy GPU). Ảnh kiểm tra nằm trong `data/viewer-verification/`. Kiểm tra ngày 15/09/2026: build TypeScript/Vite đạt; Chrome không có lỗi JavaScript; các chế độ và sự kiện hoàn tất đạt. Có kiểm tra trực quan ảnh splat, point cloud, camera và depth; đây chưa phải benchmark chất lượng toàn scene.
