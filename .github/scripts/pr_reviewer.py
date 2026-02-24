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
            logger.error(f"GLM API choice is not a dict: {type(first_choice)}")
            return None

        if "message" not in first_choice:
            logger.error("GLM API choice missing 'message' field")
            logger.error(f"Choice structure: {json.dumps(first_choice, indent=2)}")  # noqa: E501
            return None

        message = first_choice["message"]
        if not isinstance(message, dict):
            logger.error(f"GLM API message is not a dict: {type(message)}")
            return None

        if "content" not in message:
            logger.error("GLM API message missing 'content' field")
            logger.error(f"Message structure: {json.dumps(message, indent=2)}")
            return None

        content = message["content"]
        if not isinstance(content, str):
            logger.error(f"GLM API content is not a string: {type(content)}")
            return None

        return content

    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Failed to parse GLM API response: {e}")
        logger.error(f"Response: {json.dumps(data, indent=2)}")
        return None


def call_glm_api(prompt: str, api_key: str, config: Config) -> str | None:
    """
    Call the GLM-4.7 API to get code review with retry logic.

    Args:
        prompt: The prompt to send to the API.
        api_key: The GLM API key.
        config: Configuration object.

    Returns:
        The review content as a string, or None if all retries failed.
    """
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": config.glm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 32000,
        "top_p": 0.9,
    }

    last_error: str | None = None

    for attempt in range(config.max_retries):
        try:
            logger.info(f"Calling GLM API ({config.glm_model}) - Attempt {attempt + 1}/{config.max_retries}")
            response = requests.post(
                config.glm_api_url,
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code == 200:
                data = response.json()
                review = validate_glm_response(data)
                if review:
                    logger.info(f"Successfully received review ({len(review)} chars)")  # noqa: E501
                    return review
                else:
                    last_error = "Failed to validate API response structure"
                    logger.error(last_error)
            else:
                last_error = f"GLM API request failed: {response.status_code} - {response.text}"
                logger.error(last_error)

        except requests.exceptions.Timeout:
            last_error = f"GLM API request timed out (attempt {attempt + 1})"
            logger.error(last_error)
        except requests.exceptions.RequestException as e:
            last_error = f"GLM API request exception: {e}"
            logger.error(last_error)
        except json.JSONDecodeError as e:
            last_error = f"Failed to decode GLM API response as JSON: {e}"
            logger.error(last_error)

        # Exponential backoff before retry
        if attempt < config.max_retries - 1:
            wait_time: float = config.retry_delay * (2**attempt)
            logger.info(f"Waiting {wait_time}s before retry...")  # noqa: E501
            time.sleep(wait_time)

    logger.error(
        f"All {config.max_retries} attempts failed. Last error: {last_error}"  # noqa: E501
    )
    return None


def check_cooldown(  # noqa: E501
    repo_name: str,
    pr_number: int,
    token: str,
    cooldown_minutes: int,
) -> bool:
    """
    Check if the PR is within the cooldown period.

    Args:
        repo_name: The repository name in format "owner/repo".
        pr_number: The pull request number.
        token: GitHub authentication token.
        cooldown_minutes: Cooldown period in minutes.

    Returns:
        True if within cooldown period (should skip), False otherwise.
    """
    from datetime import datetime, timedelta

    url: str = (
        f"https://api.github.com/repos/{repo_name}/issues/"
        f"{pr_number}/comments"  # noqa: E501
    )
    headers: dict[str, str] = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    logger.info(f"Checking cooldown period ({cooldown_minutes} minutes)")
    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code != 200:
        logger.warning(f"Failed to check cooldown: {response.status_code}, proceeding with review")
        return False

    comments: list[dict[str, Any]] = response.json()
    cutoff_time: datetime = datetime.utcnow() - timedelta(minutes=cooldown_minutes)  # noqa: E501

    for comment in reversed(comments):
        # Check if this is a bot comment
        body: str = comment.get("body", "")
        if "VLLM-Omni PR Review" in body or "PR Reviewer Bot" in body:
            created_at_str: str = comment.get("created_at", "")
            try:
                # Parse GitHub timestamp format
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                created_at = created_at.replace(tzinfo=None)
                if created_at > cutoff_time:
                    logger.info(f"PR is within cooldown period (last review: {created_at_str})")
                    return True
            except ValueError:
                logger.warning(f"Failed to parse comment timestamp: {created_at_str}")  # noqa: E501
                continue

    logger.info("PR is outside cooldown period, proceeding with review")
    return False


def post_review_comment(  # noqa: E501
    repo_name: str,
    pr_number: int,
    token: str,
    review: str,
) -> bool:
    """
    Post the review as a comment on the PR.

    Args:
        repo_name: The repository name in format "owner/repo".
        pr_number: The pull request number.
        token: GitHub authentication token.
        review: The review content to post.

    Returns:
        True if posting succeeded, False otherwise.
    """
    url: str = (
        f"https://api.github.com/repos/{repo_name}/issues/"
        f"{pr_number}/comments"  # noqa: E501
    )
    headers: dict[str, str] = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Format the review comment
    comment_body: str = f"""## 🤖 VLLM-Omni PR Review

{review}

---
*This review was generated automatically by the VLLM-Omni PR Reviewer Bot
using {os.getenv("GLM_MODEL", DEFAULT_GLM_MODEL)}.*
"""

    payload: dict[str, str] = {"body": comment_body}

    logger.info(f"Posting review comment to PR #{pr_number}")
    response = requests.post(url, headers=headers, json=payload, timeout=30)

    if response.status_code == 201:
        logger.info("Successfully posted review comment")
        return True
    else:
        logger.error(f"Failed to post comment: {response.status_code}")
        logger.error(f"Response: {response.text}")
        return False


def main() -> int:
    """
    Main entry point for the PR reviewer bot.

    Returns:
        0 on success, 1 on error.
    """
    logger.info("VLLM-Omni PR Reviewer Bot starting...")

    # Load configuration
    config: Config = get_config()
    logger.info(
        f"Configuration: model={config.glm_model}, "
        f"cooldown={config.cooldown_minutes}min, "
        f"max_retries={config.max_retries}"
    )

    # Get environment variables
    token: str = get_env_var("GITHUB_TOKEN")
    api_key: str = get_env_var("GLM_API_KEY")
    repo_name: str = get_env_var("REPO_NAME")
    pr_number_str: str = get_env_var("PR_NUMBER")
    comment_body: str = get_env_var("COMMENT_BODY")

    try:
        pr_number: int = int(pr_number_str)
    except ValueError:
        logger.error(f"Invalid PR number: {pr_number_str}")
        return 1

    logger.info(f"Repository: {repo_name}")
    logger.info(f"PR Number: {pr_number}")

    # Check if the comment contains the trigger phrase
    if not check_trigger(comment_body):
        logger.info(
            f"Comment does not contain trigger phrase '{TRIGGER_PHRASE}', exiting"  # noqa: E501
        )
        return 0

    logger.info("Trigger phrase detected! Starting review process...")

    # Check cooldown period
    if check_cooldown(repo_name, pr_number, token, config.cooldown_minutes):
        logger.info("Skipping review due to cooldown period")
        return 0

    # Fetch PR details
    logger.info("Step 1/4: Fetching PR details...")
    pr_details: PRDetails | None = fetch_pr_details(repo_name, pr_number, token)  # noqa: E501
    if not pr_details:
        logger.error("Failed to fetch PR details")
        return 1

    pr_title: str = pr_details.get("title", "Unknown")
    pr_description: str = pr_details.get("body", "")

    logger.info(f"PR Title: {pr_title}")

    # Fetch PR diff
    logger.info("Step 2/4: Fetching PR diff...")
    diff: str | None = fetch_pr_diff(repo_name, pr_number, token, config.max_diff_size)
    if diff is None:
        logger.error("Failed to fetch PR diff")
        return 1

    if not diff:
        logger.warning("Warning: Empty diff - this might be a draft PR or no code changes")

    # Build prompt
    logger.info("Step 3/4: Building review prompt...")
    prompt: str = build_review_prompt(pr_title, pr_description, diff)

    # Call GLM API
    logger.info("Step 4/4: Calling GLM API...")
    review: str | None = call_glm_api(prompt, api_key, config)
    if not review:
        logger.error("Failed to get review from GLM API")
        return 1

    # Post review comment
    logger.info("Posting review comment...")
    if not post_review_comment(repo_name, pr_number, token, review):
        logger.error("Failed to post review comment")
        return 1

    logger.info("PR review completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
