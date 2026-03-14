# VLLM-Omni PR Reviewer

## Overview

The VLLM-Omni PR Reviewer is an automated code review bot powered by GLM-4.7 AI model. It helps maintain code quality by providing intelligent feedback on pull requests.

## Features

- **Intelligent Code Analysis**: Leverages GLM-4.7 for understanding code context and providing meaningful feedback
- **Comprehensive Reviews**: Covers code quality, architecture, security, testing, and documentation
- **Structured Output**: Provides well-formatted reviews with clear sections and actionable suggestions
- **Rate Limiting**: Built-in cooldown mechanism to prevent excessive API usage
- **Retry Logic**: Automatic retries with exponential backoff for transient API failures
- **Defensive Parsing**: Robust validation of API responses to handle malformed data
- **Cost Control**: Only repository members/collaborators/owners can trigger reviews

## How to Use

### Triggering a Review

To trigger an automated PR review, mention the bot in a PR comment:

```
@vllm-omni-reviewer please review
```

Or include in your PR description:

```
@vllm-omni-reviewer
```

The bot will automatically review your changes and post a detailed comment.

## What Gets Reviewed

- **vLLM Architecture Compatibility**: Ensures changes align with vLLM's design patterns
- **Multi-modal Integration**: Reviews audio, vision, and text processing implementations
- **Performance Implications**: Analyzes impact on inference latency and throughput
- **Code Quality**: Checks Python best practices, type hints, and documentation
- **Security Considerations**: Identifies potential security vulnerabilities
- **Testing Coverage**: Recommends additional test cases when needed

## Review Output

The bot posts a structured review comment with:

- **Overview**: Brief summary of the PR's purpose
- **Critical Issues (Must Fix)**: Blocking issues that need to be addressed
- **Important Issues (Should Fix)**: Significant concerns that should be resolved
- **Minor Issues & Suggestions**: Small improvements and optional suggestions
- **Positive Aspects**: Highlights well-implemented features
- **Performance Considerations**: Analysis of performance impact
- **Testing Recommendations**: Suggestions for additional tests
- **Overall Assessment**: Final recommendation (Approve/Request Changes/Needs Major Work)

## Rate Limiting and Cooldown

The bot includes a cooldown mechanism to prevent excessive API usage:

- **Default cooldown**: 5 minutes between reviews per PR
- **Configurable**: Can be adjusted via `PR_REVIEWER_COOLDOWN_MINUTES` environment variable
- **Smart detection**: Checks for previous bot comments before starting a review

If you trigger a review within the cooldown period, the bot will log a message and skip the review.

## Architecture

```
┌─────────────────┐
│  PR Comment     │
│  @vllm-omni-    │
│  reviewer       │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  GitHub Actions Workflow        │
│  (.github/workflows/            │
│   pr-reviewer.yml)              │
│                                 │
│  - Python 3.11                  │
│  - requests==2.31.0             │
│  - pyyaml==6.0.1                │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  PR Reviewer Script             │
│  (.github/scripts/              │
│   pr_reviewer.py)               │
│                                 │
│  1. Check cooldown              │
│  2. Fetch PR details & diff     │
│  3. Build review prompt         │
│  4. Call GLM-4.7 API            │
│     (with retry logic)          │
│  5. Validate response           │
│  6. Post review comment         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  GLM-4.7 API                    │
│  (open.bigmodel.cn)             │
└─────────────────────────────────┘
```

1. **GitHub Actions Workflow** (`.github/workflows/pr-reviewer.yml`): Triggers on @mention
2. **Python Script** (`.github/scripts/pr_reviewer.py`): Fetches PR data and calls GLM-4.7 API
3. **GLM-4.7 API**: Provides intelligent code analysis

## Testing

### Testing the PR Reviewer Bot

To test the PR reviewer bot before deploying to production:

1. **Create a test PR** - Make a small, safe change (e.g., documentation update)
2. **Open the PR** - Create a pull request with a descriptive title
3. **Trigger the review** - Comment `@vllm-omni-reviewer` on the PR
4. **Monitor results** - Check the Actions tab for workflow execution logs

### Running Unit Tests

The bot includes comprehensive unit tests that can be run locally:

```bash
# Run all tests
pytest .github/tests/test_pr_reviewer.py -v

# Run specific test
pytest .github/tests/test_pr_reviewer.py::TestCheckTrigger -v

# Run with coverage
pytest .github/tests/test_pr_reviewer.py --cov=.github/scripts/pr_reviewer.py --cov-report=term-missing
```

### What to Look For

When testing, verify that:
- [ ] The workflow triggers on the `@vllm-omni-reviewer` comment
- [ ] The cooldown mechanism works correctly
- [ ] The GLM API call completes without errors (with retry if needed)
- [ ] A review comment is posted to the PR
- [ ] The review content is meaningful and well-structured
- [ ] The cost is within the expected range (0.50-5 CNY)

### Safe Test Changes

For testing, consider making these types of safe changes:
- Documentation updates (like adding this Testing section)
- Comment improvements
- README enhancements
- Non-functional file additions

### Example Test PR

A good test PR might:
- Update a documentation file
- Add explanatory comments
- Improve code formatting
- Fix a minor typo

These changes are safe to merge if the test is successful and won't affect functionality.

## Troubleshooting

### Bot Doesn't Respond

1. **Check permissions** - Verify you have Owner/Member/Collaborator access
2. **Check Actions tab** - Look for workflow execution and view logs
3. **Check cooldown** - If another review was posted recently, wait for the cooldown period
4. **Check API key** - Ensure `GLM_API_KEY` is configured in repository secrets

### API Errors

If the GLM API call fails:
- Check the Actions tab for detailed error logs
- Verify the `GLM_API_KEY` secret is correctly configured
- Ensure sufficient API quota is available
- The bot will automatically retry up to 3 times with exponential backoff

### Review Seems Truncated

If the review appears incomplete:
- Large diffs may be truncated at 100,000 characters
- Check the logs for truncation warnings
- Consider breaking large PRs into smaller chunks

## Configuration

### Required Secrets

The following secret must be configured in the repository settings:

- `GLM_API_KEY` - Your GLM (BigModel) API key for accessing the GLM-4.7 API

To add the secret:
1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `GLM_API_KEY`
4. Value: Your GLM API key

### Optional Configuration

The following optional environment variables can be set in the workflow file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_API_URL` | `https://open.bigmodel.cn/api/paas/v4/chat/completions` | GLM API endpoint |
| `GLM_MODEL` | `glm-4.7` | Model to use for reviews |
| `PR_REVIEWER_COOLDOWN_MINUTES` | `5` | Cooldown period between reviews |
| `PR_REVIEWER_MAX_RETRIES` | `3` | Maximum API retry attempts |
| `PR_REVIEWER_RETRY_DELAY` | `1.0` | Base delay for retry backoff (seconds) |
| `PR_REVIEWER_MAX_DIFF_SIZE` | `100000` | Maximum diff size before truncation |

### Workflow Customization

The workflow can be customized in `.github/workflows/pr-reviewer.yml`:
- Change Python version (default: 3.11)
- Adjust timeout value (default: 10 minutes)
- Modify trigger conditions
- Add additional dependencies

## Code Quality

The PR reviewer script follows vllm-omni coding standards:

- **Type hints**: All functions have complete type hints following mypy strict mode
- **Logging**: Uses Python's logging module for structured logging
- **Testing**: Comprehensive unit tests with pytest
- **Pre-commit**: Script is checked by pre-commit hooks (flake8)

## Cost Estimate

| Component | Cost |
|-----------|------|
| GitHub Actions (public repo) | Free |
| GLM API (glm-4.7) | ~0.50-5 CNY per PR (varies by size) |
| Total (20 PRs/month) | ~10-100 CNY/month (~$2-15 USD) |

## Contributing

To improve the PR reviewer bot:

1. Edit `.github/scripts/pr_reviewer.py` for logic changes
2. Edit `.github/workflows/pr-reviewer.yml` for workflow changes
3. Add tests to `.github/tests/test_pr_reviewer.py`
4. Run `pre-commit run --files .github/scripts/pr_reviewer.py` to check code quality
5. Test thoroughly with a test PR before deploying to production

## License

This bot is part of the VLLM-Omni project and follows the same license terms.
