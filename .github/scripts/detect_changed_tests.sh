#!/bin/bash
# Script to detect changed files and find corresponding test files
# Usage: ./detect_changed_tests.sh [base_sha] [head_sha] [event_type]
#
# Outputs to GITHUB_OUTPUT:
#   test_files: Space-separated list of test files to run
#   has_tests: 'true' if test files found, 'false' otherwise

set -e

BASE_SHA="${1:-}"
HEAD_SHA="${2:-}"
EVENT_TYPE="${3:-push}"

# Determine base and head SHA based on event type
if [ "$EVENT_TYPE" = "pull_request" ]; then
    BASE_SHA="${BASE_SHA:-$GITHUB_BASE_REF}"
    HEAD_SHA="${HEAD_SHA:-$GITHUB_SHA}"
elif [ -z "$BASE_SHA" ] || [ -z "$HEAD_SHA" ]; then
    # For push events, use git to find the previous commit
    BASE_SHA="${BASE_SHA:-$(git rev-parse HEAD~1 2>/dev/null || echo HEAD)}"
    HEAD_SHA="${HEAD_SHA:-HEAD}"
fi

echo "Event type: $EVENT_TYPE"
echo "Base SHA: $BASE_SHA"
echo "Head SHA: $HEAD_SHA"

# Get changed Python files
CHANGED_FILES=$(git diff --name-only --diff-filter=ACMRT "$BASE_SHA" "$HEAD_SHA" | grep -E '\.(py)$' || true)

if [ -z "$CHANGED_FILES" ]; then
    echo "No Python files changed"
    echo "test_files=" >> "$GITHUB_OUTPUT"
    echo "has_tests=false" >> "$GITHUB_OUTPUT"
    exit 0
fi

echo "Changed Python files:"
echo "$CHANGED_FILES"

# Extract related test files
TEST_FILES=""
for file in $CHANGED_FILES; do
    # If changed file is a test file, add it directly
    if [[ "$file" == tests/* ]]; then
        TEST_FILES="$TEST_FILES $file"
    # If changed file is source code, find corresponding test file
    elif [[ "$file" == vllm_omni/* ]]; then
        # Convert vllm_omni/path/to/module.py to possible test file paths
        REL_PATH=$(echo "$file" | sed 's|^vllm_omni/||' | sed 's|\.py$||')
        MODULE_NAME=$(basename "$REL_PATH")
        DIR_PATH=$(dirname "$REL_PATH")

        # Try multiple possible test file naming patterns
        POSSIBLE_TESTS=(
            "tests/test_${REL_PATH//\//_}.py"
            "tests/${REL_PATH//\//_}_test.py"
        )

        # If multi-level directory, also try directory structure
        if [ "$DIR_PATH" != "." ]; then
            POSSIBLE_TESTS+=(
                "tests/${DIR_PATH}/test_${MODULE_NAME}.py"
                "tests/${DIR_PATH}/${MODULE_NAME}_test.py"
            )
        fi

        for test_file in "${POSSIBLE_TESTS[@]}"; do
            if [ -f "$test_file" ]; then
                TEST_FILES="$TEST_FILES $test_file"
                break
            fi
        done
    fi
done

# Deduplicate and format
TEST_FILES=$(echo "$TEST_FILES" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ' | sed 's/^ *//;s/ *$//')

if [ -z "$TEST_FILES" ]; then
    echo "No related test files found, will run all tests"
    echo "test_files=" >> "$GITHUB_OUTPUT"
    echo "has_tests=false" >> "$GITHUB_OUTPUT"
else
    echo "Found test files: $TEST_FILES"
    echo "test_files=$TEST_FILES" >> "$GITHUB_OUTPUT"
    echo "has_tests=true" >> "$GITHUB_OUTPUT"
fi
