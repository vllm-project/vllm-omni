import os
import time
import tempfile
import pathlib
import hashlib
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

def test_local_file_cache_key_invalidation():
    print("Testing local file cache key invalidation...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        test_file = tmp_path / "test_audio.wav"
        
        # 1. Create a dummy file and get its cache key
        test_file.write_text("dummy audio data 1")
        file_uri = f"file://{test_file}"
        
        key1 = OmniOpenAIServingSpeech._get_ref_audio_cache_key(file_uri)
        print(f"  [1] Key for original file: {key1}")
        
        # Wait slightly to ensure the mtime changes on the filesystem
        time.sleep(0.01)
        
        # 2. Modify the file
        test_file.write_text("dummy audio data 2 modified")
        
        # 3. Get the new cache key
        key2 = OmniOpenAIServingSpeech._get_ref_audio_cache_key(file_uri)
        print(f"  [2] Key for modified file: {key2}")
        
        assert key1 != key2, "Cache key should change when file is modified!"
        print("  ✓ Passed: Keys are correctly invalidated based on file metadata.")

def test_remote_url_cache_key():
    print("\nTesting remote URL cache key stability...")
    url = "https://example.com/audio.wav"
    key1 = OmniOpenAIServingSpeech._get_ref_audio_cache_key(url)
    key2 = OmniOpenAIServingSpeech._get_ref_audio_cache_key(url)
    
    assert key1 == key2, "Cache key for URLs should remain constant!"
    print("  ✓ Passed: URLs correctly maintain a constant cache key.")

def test_missing_file_cache_key():
    print("\nTesting missing file fallback...")
    file_uri = "file:///path/to/nonexistent/file.wav"
    key = OmniOpenAIServingSpeech._get_ref_audio_cache_key(file_uri)
    expected_key = hashlib.sha1(file_uri.encode("utf-8")).hexdigest()
    
    assert key == expected_key, "Missing files should fallback to the string hash."
    print("  ✓ Passed: Missing files are handled gracefully.")

if __name__ == "__main__":
    test_local_file_cache_key_invalidation()
    test_remote_url_cache_key()
    test_missing_file_cache_key()
    print("\nAll unit tests passed successfully!")
