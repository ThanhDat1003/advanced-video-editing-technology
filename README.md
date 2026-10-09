# Transcript Video tiếng Việt

CLI Windows xử lý video local bằng NVIDIA GPU, Faster Whisper `large-v3`, có checkpoint/resume và xuất TXT, SRT, JSON.

## 1. Cài môi trường

Yêu cầu máy đích:

- Windows 10/11 x64, NVIDIA RTX 12 GB và driver NVIDIA mới.
- Python 3.11 x64.
- FFmpeg và FFprobe trong `PATH`.
- CUDA Toolkit 12.x, cuBLAS và cuDNN 9 cho CUDA 12 trong `PATH`.
- RAM tối thiểu 16 GB; nên có ít nhất 15 GB ổ đĩa trống.

Kiểm tra:

```powershell
py -3.11 --version
ffmpeg -version
ffprobe -version
nvidia-smi
```

Tạo môi trường và cài tool:

```powershell
cd "D:\Công việc\Projects\transcipt-video"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock
pip install -e ".[test]" --no-deps
```

Hoặc, sau khi đã cài Python/FFmpeg/CUDA/cuDNN và thêm chúng vào `PATH`, chạy script thiết lập:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup-windows.ps1
```

Lần chạy đầu cần Internet để tải `large-v3`. Các lần sau dùng model trong cache và có thể offline.

## 2. Sử dụng

```powershell
transcript-video transcribe "D:\Videos\bai-giang.mp4" --output "D:\Transcripts"
```

Nhiều video:

```powershell
transcript-video transcribe "D:\Videos\bai-1.mp4" "D:\Videos\bai-2.mkv" -o "D:\Transcripts"
```

Cả một thư mục (tự quét file .mp4, .mkv, .mov, .avi, .wmv, .flv, .webm, .m4v, .ts, .mts, .m2ts nằm trực tiếp trong thư mục, không đệ quy vào thư mục con):

```powershell
transcript-video transcribe "D:\Videos" -o "D:\Transcripts"
```

Glossary UTF-8, mỗi dòng một thuật ngữ:

```powershell
transcript-video transcribe video.mp4 -o transcripts --glossary glossary.txt
```

Tool mặc định resume. Nếu kết quả cuối đã tồn tại, tool không ghi đè:

```powershell
transcript-video transcribe video.mp4 -o transcripts --overwrite
```

CPU fallback phải được yêu cầu rõ ràng:

```powershell
transcript-video transcribe video.mp4 -o transcripts --device cpu
```

Để kéo-thả, thả video vào `transcribe-video.bat`. Batch file tự dùng `.venv` cạnh nó; kết quả mặc định nằm trong `transcripts\<tên-video>`.

## 3. Kết quả

Mỗi video tạo một thư mục gồm:

- `transcript.txt`
- `transcript.srt`
- `transcript.json`
- `checkpoint.json`
- `run.log`

Audio WAV tạm được xóa sau mỗi chunk. `checkpoint.json` được ghi atomically sau từng chunk khoảng 15 phút.

## 4. Khắc phục lỗi

- **Không tìm thấy CUDA:** kiểm tra CUDA 12/cuDNN 9 DLL có trong `PATH`, sau đó mở terminal mới.
- **CUDA out of memory:** đóng ứng dụng dùng GPU và chạy lại cùng lệnh; resume tiếp tục từ checkpoint. Tool tự giảm batch 4 → 2 → 1.
- **Checkpoint không hợp lệ:** file nguồn hoặc cấu hình đã đổi. Dùng output khác hoặc `--overwrite` nếu muốn chạy lại.
- **Nhiều audio track:** xem stream bằng `ffprobe`, rồi truyền `--audio-stream INDEX`.
