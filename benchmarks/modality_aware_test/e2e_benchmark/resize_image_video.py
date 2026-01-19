import argparse
import multiprocessing
import os
import subprocess

from PIL import Image
from tqdm import tqdm

IMAGE_TARGET_SIZE = (512, 512)
VIDEO_TARGET_SIZE = (256, 256)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def resize_image(file_path):
    try:
        with Image.open(file_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            img_resized = img.resize(IMAGE_TARGET_SIZE, Image.Resampling.LANCZOS)

            img_resized.save(file_path, quality=95)
        return True, None
    except Exception as e:
        return False, f"[Image Error] {file_path}: {e}"


def resize_video(file_path):
    temp_path = file_path + ".tmp.mp4"
    try:
        w, h = VIDEO_TARGET_SIZE
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            file_path,
            "-vf",
            f"scale={w}:{h},setsar=1:1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "copy",
            temp_path,
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        os.replace(temp_path, file_path)
        return True, None
    except subprocess.CalledProcessError:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, f"[Video ffmpeg Error] {file_path}"
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, f"[Video Error] {file_path}: {e}"


def worker(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in IMAGE_EXTS:
        return resize_image(file_path)
    elif ext in VIDEO_EXTS:
        return resize_video(file_path)
    else:
        return True, None


def main():
    parser = argparse.ArgumentParser(description="Resize all images and videos in a directory to fixed resolutions.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument(
        "--workers", type=int, default=min(16, multiprocessing.cpu_count()), help="Number of parallel workers"
    )
    args = parser.parse_args()

    target_files = []

    print(f"🔍 Scanning '{args.data_dir}' for media files...")

    for root, _, files in os.walk(args.data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                target_files.append(os.path.join(root, file))

    if not target_files:
        print("⚠️ No image or video files found.")
        return

    print(f"✅ Found {len(target_files)} files.")
    print(f"   - Image Target: {IMAGE_TARGET_SIZE}")
    print(f"   - Video Target: {VIDEO_TARGET_SIZE}")
    print(f"🚀 Processing with {args.workers} workers...")
    print("⏳ This may take a while depending on video length...")

    errors = []
    with multiprocessing.Pool(args.workers) as pool:
        for success, error_msg in tqdm(pool.imap_unordered(worker, target_files), total=len(target_files)):
            if not success:
                errors.append(error_msg)

    print("\n" + "=" * 50)
    print("🎉 Processing Complete!")
    print(f"✅ Success: {len(target_files) - len(errors)}")
    print(f"❌ Failed:  {len(errors)}")

    if errors:
        print("\nError Report:")
        for err in errors:
            print(err)
    print("=" * 50)


if __name__ == "__main__":
    main()
