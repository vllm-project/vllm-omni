import os
import random
import subprocess
import argparse
import sys
from tqdm import tqdm

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("❌ Error: 'ffmpeg' command not found. Please install FFmpeg (e.g., 'apt-get install ffmpeg').")
        sys.exit(1)

def truncate_videos(data_dir):
    # Inferring path based on your previous script: [data_dir]/video/files
    video_dir = os.path.join(data_dir, "video", "files")

    if not os.path.exists(video_dir):
        print(f"❌ Error: Video directory not found: {video_dir}")
        print("   Please ensure you have downloaded the video dataset first.")
        return

    # Gather all mp4 files
    files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    total_files = len(files)
    
    if total_files == 0:
        print(f"⚠️ No .mp4 files found in {video_dir}")
        return

    print(f"🚀 Found {total_files} videos in {video_dir}")
    print("✂️  Starting truncation (Target: 2.0s - 3.0s)...")

    # Initialize stats
    processed_count = 0
    error_count = 0

    pbar = tqdm(files, desc="Processing")
    for filename in pbar:
        full_path = os.path.join(video_dir, filename)
        temp_path = os.path.join(video_dir, f"temp_{filename}")

        # Generate a random duration between 2.0 and 3.0 seconds
        target_duration = random.uniform(2.0, 3.0)

        # FFmpeg command construction
        # -y: Overwrite output files without asking
        # -i: Input file
        # -ss: Start time (00:00:00)
        # -t: Duration (random value)
        # -c:v libx264: Re-encode video (ensure keyframes are correct after cut)
        # -preset ultrafast: Prioritize speed over compression ratio
        # -c:a copy: Copy audio stream directly (fast)
        # -loglevel error: Suppress verbose output
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", full_path,
            "-ss", "0",
            "-t", f"{target_duration:.2f}",
            "-c:v", "libx264", 
            "-preset", "ultrafast",
            "-c:a", "copy",
            temp_path
        ]

        try:
            # Run FFmpeg
            subprocess.run(cmd, check=True)
            
            # If successful, replace the original file with the truncated one
            # os.replace is atomic on POSIX systems
            os.replace(temp_path, full_path)
            processed_count += 1
            
        except subprocess.CalledProcessError as e:
            error_count += 1
            # Clean up temp file if it was created but failed
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            error_count += 1
            print(f"\nError processing {filename}: {e}")

    pbar.close()
    
    print("\n✅ Truncation complete!")
    print(f"   - Processed: {processed_count}")
    print(f"   - Errors:    {error_count}")
    print(f"   - Target directory: {video_dir}")

if __name__ == "__main__":

    # python3 vllm-omni/benchmarks/modality_aware_scheduling/truncate_video.py --data_dir /root/data/datasets
    parser = argparse.ArgumentParser(description="Truncate existing video dataset to random 2s-3s lengths")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory where datasets were downloaded")
    
    args = parser.parse_args()
    
    check_ffmpeg()
    truncate_videos(args.data_dir)