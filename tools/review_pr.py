#!/usr/bin/env python3
"""
PR Review Tool for vLLM-Omni
===========================

This script helps review Pull Requests for the vLLM-omni project from an AI expert perspective.

Usage:
    python tools/review_pr.py --pr-number 19
    python tools/review_pr.py --pr-number 19 --detailed
    python tools/review_pr.py --pr-number 19 --export review_output.md

Requirements:
    - GitHub CLI (gh) installed and authenticated
    - OR GitHub token set in GITHUB_TOKEN environment variable
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


class PRReviewer:
    """AI Expert PR Reviewer for vLLM-Omni"""
    
    def __init__(self, pr_number: int, repo: str = "hsliuustc0106/vllm-omni"):
        self.pr_number = pr_number
        self.repo = repo
        self.pr_data = None
        self.files_changed = []
        
    def fetch_pr_data(self) -> bool:
        """Fetch PR data using GitHub CLI"""
        try:
            cmd = [
                "gh", "pr", "view", str(self.pr_number),
                "--repo", self.repo,
                "--json", "title,body,state,author,additions,deletions,files,commits,reviews"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.pr_data = json.loads(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error fetching PR data: {e}")
            print(f"Make sure 'gh' is installed and you're authenticated")
            return False
        except json.JSONDecodeError as e:
            print(f"Error parsing PR data: {e}")
            return False
    
    def fetch_diff(self) -> bool:
        """Fetch PR diff"""
        try:
            cmd = ["gh", "pr", "diff", str(self.pr_number), "--repo", self.repo]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.diff = result.stdout
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error fetching diff: {e}")
            return False
    
    def analyze_changes(self) -> Dict[str, any]:
        """Analyze the PR changes from an AI perspective"""
        if not self.pr_data:
            return {}
        
        analysis = {
            "summary": {
                "title": self.pr_data.get("title", ""),
                "state": self.pr_data.get("state", ""),
                "author": self.pr_data.get("author", {}).get("login", ""),
                "additions": self.pr_data.get("additions", 0),
                "deletions": self.pr_data.get("deletions", 0),
                "files_changed": len(self.pr_data.get("files", [])),
            },
            "areas_of_concern": [],
            "positive_aspects": [],
            "recommendations": [],
        }
        
        # Analyze file changes
        files = self.pr_data.get("files", [])
        
        # Check for critical files
        critical_areas = {
            "engine": [],
            "model_executor": [],
            "worker": [],
            "core": [],
            "config": [],
        }
        
        for file_info in files:
            path = file_info.get("path", "")
            for area, files_list in critical_areas.items():
                if area in path:
                    files_list.append(path)
        
        # Generate concerns based on changes
        if critical_areas["engine"]:
            analysis["areas_of_concern"].append({
                "area": "Engine Modifications",
                "files": critical_areas["engine"],
                "concern": "Changes to engine components require careful review for performance and correctness",
                "checks": [
                    "Verify batching logic is correct",
                    "Check cache management",
                    "Ensure no memory leaks",
                    "Validate scheduling changes"
                ]
            })
        
        if critical_areas["model_executor"]:
            analysis["areas_of_concern"].append({
                "area": "Model Execution",
                "files": critical_areas["model_executor"],
                "concern": "Model executor changes affect inference correctness",
                "checks": [
                    "Verify tensor shapes",
                    "Check numerical precision",
                    "Validate model loading",
                    "Test with different model sizes"
                ]
            })
        
        if critical_areas["worker"]:
            analysis["areas_of_concern"].append({
                "area": "Worker Changes",
                "files": critical_areas["worker"],
                "concern": "Worker modifications impact distributed execution",
                "checks": [
                    "Check GPU memory management",
                    "Verify synchronization",
                    "Test multi-GPU scenarios",
                    "Validate worker lifecycle"
                ]
            })
        
        # Check for test coverage
        test_files = [f for f in files if "test" in f.get("path", "").lower()]
        if len(test_files) < len(files) * 0.3:
            analysis["areas_of_concern"].append({
                "area": "Test Coverage",
                "concern": f"Only {len(test_files)} test files for {len(files)} changed files",
                "checks": [
                    "Add unit tests for new functionality",
                    "Add integration tests for critical paths",
                    "Test edge cases",
                ]
            })
        else:
            analysis["positive_aspects"].append(
                f"Good test coverage: {len(test_files)} test files included"
            )
        
        # Generate recommendations
        if analysis["summary"]["additions"] > 500:
            analysis["recommendations"].append(
                "Large PR (>500 lines added). Consider splitting into smaller PRs for easier review."
            )
        
        if critical_areas["config"]:
            analysis["recommendations"].append(
                "Configuration changes detected. Ensure backward compatibility and update documentation."
            )
        
        # Check for documentation
        doc_files = [f for f in files if f.get("path", "").endswith((".md", ".rst"))]
        if not doc_files and len(files) > 5:
            analysis["recommendations"].append(
                "No documentation updates found. Consider adding/updating docs for significant changes."
            )
        
        return analysis
    
    def generate_review_report(self, detailed: bool = False) -> str:
        """Generate a comprehensive review report"""
        if not self.pr_data:
            return "Error: PR data not available"
        
        analysis = self.analyze_changes()
        
        report = f"""
# PR #{self.pr_number} Review Report - AI Expert Perspective

## Summary
**Title:** {analysis['summary']['title']}
**State:** {analysis['summary']['state']}
**Author:** {analysis['summary']['author']}
**Changes:** +{analysis['summary']['additions']} -{analysis['summary']['deletions']} lines across {analysis['summary']['files_changed']} files

## AI Expert Analysis

### Areas Requiring Careful Review
"""
        
        for concern in analysis['areas_of_concern']:
            report += f"\n#### {concern.get('area', 'Unknown Area')}\n"
            if 'concern' in concern:
                report += f"**Concern:** {concern['concern']}\n\n"
            if 'files' in concern:
                report += "**Files:**\n"
                for f in concern['files']:
                    report += f"- `{f}`\n"
            if 'checks' in concern:
                report += "\n**Review Checklist:**\n"
                for check in concern['checks']:
                    report += f"- [ ] {check}\n"
            report += "\n"
        
        report += "\n### Positive Aspects\n\n"
        if analysis['positive_aspects']:
            for aspect in analysis['positive_aspects']:
                report += f"- {aspect}\n"
        else:
            report += "_None identified from automated analysis. Review code for specific positives._\n"
        
        report += "\n### Recommendations\n\n"
        for rec in analysis['recommendations']:
            report += f"- {rec}\n"
        
        report += """
## Review Process

### 1. Architecture Review
- [ ] Changes align with vLLM-omni architecture
- [ ] No unnecessary coupling introduced
- [ ] Proper abstraction and modularity
- [ ] Consistent with existing patterns

### 2. Performance Review
- [ ] No performance regressions
- [ ] Memory usage acceptable
- [ ] GPU utilization optimized
- [ ] Benchmarks provided (if applicable)

### 3. Code Quality Review
- [ ] Code follows PEP 8
- [ ] Proper type hints
- [ ] Comprehensive docstrings
- [ ] Clear variable/function names
- [ ] No code duplication

### 4. Testing Review
- [ ] Unit tests cover new code
- [ ] Integration tests for workflows
- [ ] Edge cases tested
- [ ] Tests are maintainable

### 5. ML/AI Specifics
- [ ] Numerical stability ensured
- [ ] Correct precision handling
- [ ] Model compatibility verified
- [ ] Inference correctness validated

### 6. Documentation Review
- [ ] API documentation updated
- [ ] README updated if needed
- [ ] Examples provided
- [ ] Breaking changes documented

## Next Steps

1. Review the specific code changes
2. Check each item in the review checklist
3. Run tests locally
4. Test with sample models
5. Provide inline comments on GitHub
6. Approve or request changes

---
**Reviewer Note:** This is an automated AI expert analysis. Manual review of code is still required.
"""
        
        return report
    
    def export_report(self, filename: str):
        """Export the review report to a file"""
        report = self.generate_review_report(detailed=True)
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Review report exported to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Review vLLM-Omni PRs from an AI expert perspective"
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        required=True,
        help="Pull request number to review"
    )
    parser.add_argument(
        "--repo",
        default="hsliuustc0106/vllm-omni",
        help="Repository (owner/name)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed analysis"
    )
    parser.add_argument(
        "--export",
        help="Export report to file"
    )
    
    args = parser.parse_args()
    
    print(f"Fetching PR #{args.pr_number} from {args.repo}...")
    
    reviewer = PRReviewer(args.pr_number, args.repo)
    
    if not reviewer.fetch_pr_data():
        sys.exit(1)
    
    print("Analyzing PR changes...")
    
    if args.export:
        reviewer.export_report(args.export)
    else:
        report = reviewer.generate_review_report(args.detailed)
        print(report)


if __name__ == "__main__":
    main()
