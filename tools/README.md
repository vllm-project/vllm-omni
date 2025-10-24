# Tools Directory

This directory contains helper tools for the vLLM-omni project.

## PR Review Tool

### `review_pr.py`

A Python script that automates PR reviews from an AI expert perspective.

#### Setup

1. Install GitHub CLI:
   ```bash
   # macOS
   brew install gh
   
   # Linux
   sudo apt install gh
   # or
   conda install gh --channel conda-forge
   ```

2. Authenticate with GitHub:
   ```bash
   gh auth login
   ```

#### Usage

Basic review:
```bash
python tools/review_pr.py --pr-number 19
```

Detailed analysis:
```bash
python tools/review_pr.py --pr-number 19 --detailed
```

Export to file:
```bash
python tools/review_pr.py --pr-number 19 --export PR_19_review_output.md
```

#### What It Does

The tool analyzes PRs for:
- Changes to critical components (engine, workers, model executor)
- Test coverage
- Documentation updates
- Performance implications
- ML/AI specific concerns

It generates a comprehensive review report with:
- Summary of changes
- Areas requiring careful review
- Positive aspects
- Actionable recommendations
- Review checklist

#### Limitations

- Requires GitHub authentication
- Works best with public repositories or when you have access to the repo
- Automated analysis should be supplemented with manual code review

##Future Tools

Additional tools to be added:
- Model compatibility checker
- Performance benchmark comparator
- Configuration validator
- Memory profiler wrapper
