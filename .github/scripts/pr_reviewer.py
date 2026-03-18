#!/usr/bin/env python3
"""
PR Reviewer using GLM API for vllm-omni project.
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, TypedDict

import requests


# Type definitions for API responses
class PRDetails(TypedDict):
    """Type definition for GitHub PR details response."""

    title: str
    body: str
    number: int
    state: str
    user: dict[str, Any]


class GLMMessage(TypedDict):
    """Type definition for GLM API message."""

    role: str
    content: str


class GLMChoice(TypedDict):
    """Type definition for GLM API choice."""

    message: GLMMessage
    finish_reason: str


class GLMResponse(TypedDict):
    """Type definition for GLM API response."""

    choices: list[GLMChoice]
    usage: dict[str, int] | None


class GitHubComment(TypedDict):
    """Type definition for GitHub comment."""

    id: int
    body: str
    created_at: str
    user: dict[str, Any]


# Configuration
TRIGGER_PHRASE: str = "@vllm-omni-reviewer"
DEFAULT_GLM_API_URL: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"  # noqa: E501
DEFAULT_GLM_MODEL: str = "glm-5"
DEFAULT_COOLDOWN_MINUTES: int = 5
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_DELAY: float = 1.0
MAX_DIFF_SIZE: int = 100_000  # Maximum diff size in characters


@dataclass
class Config:
    """Configuration for the PR reviewer."""

    glm_api_url: str
    glm_model: str
    cooldown_minutes: int
    max_retries: int
    retry_delay: float
    max_diff_size: int


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[PR Reviewer] %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)


def get_config() -> Config:
    """Load configuration from environment variables with defaults."""
    return Config(
        glm_api_url=os.getenv("GLM_API_URL", DEFAULT_GLM_API_URL),
        glm_model=os.getenv("GLM_MODEL", DEFAULT_GLM_MODEL),
        cooldown_minutes=int(
            os.getenv(
                "PR_REVIEWER_COOLDOWN_MINUTES",
                str(DEFAULT_COOLDOWN_MINUTES),
            )
        ),
        max_retries=int(
            os.getenv(
                "PR_REVIEWER_MAX_RETRIES",
                str(DEFAULT_MAX_RETRIES),
            )
        ),
        retry_delay=float(os.getenv("PR_REVIEWER_RETRY_DELAY", str(DEFAULT_RETRY_DELAY))),
        max_diff_size=int(os.getenv("PR_REVIEWER_MAX_DIFF_SIZE", str(MAX_DIFF_SIZE))),  # noqa: E501
    )


def get_env_var(name: str) -> str:
    """
    Get an environment variable or raise an error.

    Args:
        name: Name of the environment variable.

    Returns:
        The value of the environment variable.

    Raises:
        SystemExit: If the environment variable is not set.
    """
    value = os.environ.get(name)
    if not value:
        logger.error(f"Environment variable {name} is not set")
        sys.exit(1)
    return value


def check_trigger(comment_body: str) -> bool:
    """
    Check if the comment contains the trigger phrase.

    Args:
        comment_body: The body of the comment to check.

    Returns:
        True if the trigger phrase is found, False otherwise.
    """
    return TRIGGER_PHRASE in comment_body


def fetch_pr_diff(
    repo_name: str,
    pr_number: int,
    token: str,
    max_size: int = MAX_DIFF_SIZE,
) -> str | None:
    """
    Fetch the diff for a pull request.

    Args:
        repo_name: The repository name in format "owner/repo".
        pr_number: The pull request number.
        token: GitHub authentication token.
        max_size: Maximum diff size in characters.

    Returns:
        The diff content as a string, or None if fetching failed.
        Returns empty string if diff is larger than max_size.
    """
    url: str = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff",
    }

    logger.info(f"Fetching PR diff from {url}")
    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code == 200:
        diff: str = response.text
        if len(diff) > max_size:
            logger.warning(
                f"Diff size ({len(diff)} bytes) exceeds maximum "
                f"({max_size} bytes), truncating to first "
                f"{max_size} bytes"
            )
            return diff[:max_size] + "\n\n... [Diff truncated due to size] ..."
        logger.info(f"Successfully fetched diff ({len(diff)} bytes)")
        return diff
    else:
        logger.error(f"Failed to fetch PR diff: {response.status_code}")
        logger.error(f"Response: {response.text}")
        return None


def fetch_pr_details(
    repo_name: str,
    pr_number: int,
    token: str,
) -> PRDetails | None:
    """
    Fetch PR details including title and description.

    Args:
        repo_name: The repository name in format "owner/repo".
        pr_number: The pull request number.
        token: GitHub authentication token.

    Returns:
        A dictionary containing PR details, or None if fetching failed.
    """
    url: str = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    logger.info(f"Fetching PR details from {url}")
    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to fetch PR details: {response.status_code}")
        return None


def build_review_prompt(pr_title: str, pr_description: str, diff: str) -> str:
    """
    Build the prompt for the GLM-4.7 API.

    Args:
        pr_title: The title of the pull request.
        pr_description: The description/body of the pull request.
        diff: The diff content of the pull request.

    Returns:
        The formatted prompt string for the API.
    """
    return f"""You are an expert code reviewer for the VLLM-Omni project. \
Please review the following pull request:

## Pull Request Details
**Title:** {pr_title}

**Description:**
{pr_description if pr_description else "No description provided."}

## Code Changes (Diff)
{diff}

## Review Guidelines

Please provide a comprehensive code review with the following sections:

### 1. Overview
- Brief summary of the changes
- Overall assessment (positive, neutral, or concerns)

### 2. Code Quality
- Code style and consistency
- Potential bugs or edge cases
- Performance considerations
- Error handling

### 3. Architecture & Design
- Integration with existing codebase
- Design patterns and best practices
- Potential improvements

### 4. Security & Safety
- Security concerns (if any)
- Resource management
- Input validation

### 5. Testing & Documentation
- Test coverage considerations
- Documentation completeness
- Examples and usage clarity

### 6. Specific Suggestions
- Line-by-line specific feedback (use `file:line` format)
- Concrete actionable suggestions
- Code examples for improvements (if applicable)

### 7. Approval Status
- **LGTM** (Looks Good To Me) if the PR is ready to merge
- **LGTM with suggestions** if the PR is good but has minor suggestions
- **Changes requested** if significant changes are needed

## Important Notes
- Be constructive and helpful
- Focus on objective technical feedback
- Acknowledge good practices when you see them
- Prioritize critical issues over nitpicks
- If the diff is empty or minimal, acknowledge this and provide
  any relevant context-specific guidance

Please format your response in Markdown with clear section headers.
"""


def validate_glm_response(data: dict[str, Any]) -> str | None:
    """
    Validate and extract content from GLM API response.

    Args:
        data: The response data from GLM API.

    Returns:
        The review content string if valid, None otherwise.
    """
    # Check if choices exists and is a non-empty list
    if "choices" not in data:
        logger.error("GLM API response missing 'choices' field")
        logger.error(f"Response structure: {json.dumps(data, indent=2)}")
        return None

    choices = data["choices"]
    if not isinstance(choices, list):
        logger.error(f"GLM API 'choices' is not a list: {type(choices)}")
        return None

    if len(choices) == 0:
        logger.error("GLM API 'choices' is an empty list")
        return None

    # Check if first choice has message
    try:
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            logger.error(f"GLM API choice is not a