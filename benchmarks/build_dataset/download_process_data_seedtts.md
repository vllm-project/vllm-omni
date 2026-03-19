# Benchmark Dataset Preparation Guide

This guide describes how to download and prepare the SeedTTS test dataset for benchmarking Qwen-Omni models.

## Prerequisites

- Python 3.8+
- `gdown` for downloading from Google Drive
- Access to the benchmark scripts

## Steps

### 1. Navigate to the Dataset Directory

```bash
cd benchmarks/build_dataset
```

### 2. Install Dependencies

```bash
pip install gdown
```

### 3. Download the SeedTTS Test Dataset

Download the dataset from Google Drive:

```bash
gdown --id 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP
```

### 4. Extract the Dataset

```bash
tar -xf seedtts_testset.tar
```

### 5. Prepare the Metadata File

Copy the English metadata file to the working directory:

```bash
cp seedtts_testset/en/meta.lst meta.lst
```

### 6. Extract Prompts

Extract the first N prompts from the metadata file:

```bash
# Extract top 100 prompts (adjust -n for different amounts)
python extract_tts_prompts.py -i meta.lst -o top100.txt -n 100
```

**Options:**
- `-i, --input`: Input metadata file (default: `meta.lst`)
- `-o, --output`: Output prompts file (default: `prompts.txt`)
- `-n, --num-lines`: Number of prompts to extract (required)

### 7. Clean Up (Optional)

Remove temporary files to save disk space:

```bash
rm -rf seedtts_testset
rm seedtts_testset.tar
rm meta.lst
```

## Quick Start (All-in-One)

```bash
# Full setup and benchmark
cd benchmarks/build_dataset
pip install gdown
gdown --id 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP
tar -xf seedtts_testset.tar
cp seedtts_testset/en/meta.lst meta.lst
python extract_tts_prompts.py -i meta.lst -o top100.txt -n 100
rm -rf seedtts_testset seedtts_testset.tar meta.lst
```
