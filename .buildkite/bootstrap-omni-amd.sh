#!/bin/bash
# vLLM-Omni AMD CI Bootstrap
# Intelligent CI orchestration following vLLM's ci-infra approach
#
# Features:
# - Smart change detection (docs-only skip, critical files)
# - Pure bash YAML parsing
# - Test filtering by source_file_dependencies and mirror_hardwares
# - GitHub PR label support (ready-run-all-tests, ci-no-fail-fast)
# - Dynamic Buildkite pipeline generation

set -euo pipefail

#==============================================================================
# SECTION 1: INITIALIZATION & ENVIRONMENT DETECTION
#==============================================================================

# Enable debugging if requested
DEBUG="${VLLM_CI_DEBUG:-0}"
[[ "$DEBUG" == "1" ]] && set -x

echo "=== vLLM-Omni AMD CI Bootstrap ==="
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "Branch: ${BUILDKITE_BRANCH:-unknown}"
echo "Commit: ${BUILDKITE_COMMIT:-unknown}"
echo "Pull Request: ${BUILDKITE_PULL_REQUEST:-none}"
echo ""

# Validate environment
if [ ! -d ".buildkite" ]; then
    echo "Error: .buildkite directory not found"
    echo "Please run this script from the repository root"
    exit 1
fi

if [ ! -f ".buildkite/test-amd.yaml" ]; then
    echo "Error: .buildkite/test-amd.yaml not found"
    exit 1
fi

# Validate git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not a git repository"
    exit 1
fi

# Determine base branch for comparison
if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
    BASE_BRANCH="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
else
    BASE_BRANCH="main"
fi

echo "Base branch for comparison: ${BASE_BRANCH}"
echo ""

#==============================================================================
# SECTION 2: GITHUB LABEL CHECKING
#==============================================================================

# Function: Check GitHub PR labels
check_github_labels() {
    local pr_number="$1"

    # Extract owner/repo from git URL
    local repo_full_name=$(git remote get-url origin 2>/dev/null | sed -E 's/.*github\.com[:/]([^/]+\/[^/]+)(\.git)?$/\1/' || echo "")

    if [[ -z "$repo_full_name" ]]; then
        echo "Warning: Could not determine GitHub repository"
        return 1
    fi

    echo "--- Checking GitHub PR labels"
    echo "Repository: ${repo_full_name}"
    echo "PR Number: ${pr_number}"

    # Try to fetch labels via GitHub API (no auth needed for public repos)
    if command -v curl >/dev/null 2>&1; then
        local api_url="https://api.github.com/repos/${repo_full_name}/pulls/${pr_number}"
        local response=$(curl -s -f "${api_url}" 2>/dev/null || echo "")

        if [[ -n "$response" ]]; then
            # Extract label names
            local labels=$(echo "$response" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ',' || echo "")

            # Check for specific labels
            if [[ "$labels" == *"ready-run-all-tests"* ]]; then
                RUN_ALL_TESTS=1
                echo "✓ Found label: ready-run-all-tests"
            fi

            if [[ "$labels" == *"ci-no-fail-fast"* ]]; then
                NO_FAIL_FAST=1
                echo "✓ Found label: ci-no-fail-fast"
            fi

            [[ "$RUN_ALL_TESTS" == "0" ]] && echo "  No run-all-tests label"
            [[ "$NO_FAIL_FAST" == "0" ]] && echo "  No fail-fast override label"
            return 0
        fi
    fi

    echo "Warning: Could not fetch GitHub labels (API unavailable or request failed)"
    return 1
}

# Initialize flags
RUN_ALL_TESTS=0
NO_FAIL_FAST=0

# Check labels if this is a PR
if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]] && [[ "${BUILDKITE_PULL_REQUEST}" != "" ]]; then
    check_github_labels "${BUILDKITE_PULL_REQUEST}" || echo "Continuing without label information"
    echo ""
fi

#==============================================================================
# SECTION 3: CHANGE DETECTION & ANALYSIS
#==============================================================================

# Function: Detect if only docs changed
is_docs_only_change() {
    local changed_files="$1"

    # Docs-related patterns
    local docs_patterns=(
        "^docs/"
        "^README"
        "\\.md$"
        "\\.rst$"
        "^LICENSE"
        "^CONTRIBUTING"
    )

    local has_non_docs=0

    while IFS= read -r file; do
        [[ -z "$file" ]] && continue

        local is_docs=0
        for pattern in "${docs_patterns[@]}"; do
            if [[ "$file" =~ $pattern ]]; then
                is_docs=1
                break
            fi
        done

        if [[ "$is_docs" == "0" ]]; then
            # Found non-docs file
            has_non_docs=1
            break
        fi
    done <<< "$changed_files"

    # Return 0 (success) if all files are docs, 1 otherwise
    [[ "$has_non_docs" == "0" ]]
}

# Function: Detect critical file changes
has_critical_file_changes() {
    local changed_files="$1"

    # Critical patterns that trigger full test run
    local critical_patterns=(
        "^docker/Dockerfile"
        "^requirements.*\\.txt$"
        "^setup\\.py$"
        "^pyproject\\.toml$"
        "^\\.buildkite/bootstrap-omni-amd\\.sh$"
        "^\\.buildkite/test-amd\\.yaml$"
        "^\\.buildkite/scripts/hardware_ci/run-amd-test\\.sh$"
    )

    while IFS= read -r file; do
        [[ -z "$file" ]] && continue

        for pattern in "${critical_patterns[@]}"; do
            if [[ "$file" =~ $pattern ]]; then
                echo "  Critical file changed: $file"
                return 0
            fi
        done
    done <<< "$changed_files"

    return 1
}

echo "--- Analyzing changed files"

# Get list of changed files
CHANGED_FILES=""
if git rev-parse "origin/${BASE_BRANCH}" >/dev/null 2>&1; then
    # Fetch latest base branch
    echo "Fetching origin/${BASE_BRANCH}..."
    git fetch origin "${BASE_BRANCH}" >/dev/null 2>&1 || true

    # Get changed files between base and current commit
    CHANGED_FILES=$(git diff --name-only "origin/${BASE_BRANCH}...${BUILDKITE_COMMIT}" 2>/dev/null || \
                    git diff --name-only "origin/${BASE_BRANCH}" "${BUILDKITE_COMMIT}" 2>/dev/null || \
                    echo "")
else
    echo "Warning: Could not find base branch ${BASE_BRANCH}"
    echo "Will run all tests as a safety measure"
    RUN_ALL_TESTS=1
fi

# Count changed files
CHANGED_FILE_COUNT=$(echo "$CHANGED_FILES" | grep -c . || echo "0")
echo "Changed files: ${CHANGED_FILE_COUNT}"

# Debug: Show changed files if in debug mode
if [[ "$DEBUG" == "1" ]] && [[ -n "$CHANGED_FILES" ]]; then
    echo "Changed files list:"
    echo "$CHANGED_FILES" | head -20
    [[ "$CHANGED_FILE_COUNT" -gt 20 ]] && echo "... (${CHANGED_FILE_COUNT} total files)"
    echo ""
fi

# Check for docs-only changes (early exit optimization)
if [[ "$CHANGED_FILE_COUNT" -gt 0 ]] && is_docs_only_change "$CHANGED_FILES"; then
    echo ""
    echo "=== Documentation-only changes detected ==="
    echo "Skipping CI tests to save resources"
    echo ""

    # Generate minimal pipeline
    cat > .buildkite/pipeline.yaml << 'EOF'
steps:
  - label: ":memo: Docs-only change"
    command: echo "Only documentation changed, skipping tests"
    agents:
      queue: cpu_queue_premerge
EOF

    echo "--- Generated minimal pipeline:"
    cat .buildkite/pipeline.yaml
    echo ""

    echo "--- Uploading pipeline to Buildkite"
    buildkite-agent pipeline upload .buildkite/pipeline.yaml

    echo "=== Bootstrap Complete (docs-only) ==="
    exit 0
fi

# Check for critical file changes
if has_critical_file_changes "$CHANGED_FILES"; then
    echo "  → Critical files detected: Will run ALL tests"
    RUN_ALL_TESTS=1
fi

echo ""

#==============================================================================
# SECTION 4: YAML PARSING (Pure Bash)
#==============================================================================

echo "--- Parsing test-amd.yaml"

# Parse test-amd.yaml into structured variables
# Variables will be named: STEP_<N>_<FIELD>
# Arrays/lists use || as delimiter

TOTAL_STEPS=0
declare -A STEP_DATA

parse_test_yaml() {
    local yaml_file="$1"

    local step_num=0
    local in_step=0
    local current_section=""
    local list_indent=0

    while IFS= read -r raw_line; do
        # Handle line without removing all whitespace (need to detect indentation)
        local line="$raw_line"

        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// /}" ]] && continue

        # Detect indentation level
        local indent=$(echo "$line" | sed -E 's/^([[:space:]]*).*/\1/' | wc -c)
        indent=$((indent - 1))  # wc -c counts \n

        # Detect new step (starts with "- label:")
        if [[ "$line" =~ ^[[:space:]]*-[[:space:]]+label:[[:space:]]*(.+)$ ]]; then
            step_num=$((step_num + 1))
            in_step=1
            local label="${BASH_REMATCH[1]}"
            label=$(echo "$label" | sed 's/^["'"'"']//' | sed 's/["'"'"']$//')

            eval "STEP_${step_num}_LABEL=\"\$label\""
            eval "STEP_${step_num}_KEY=\"step-${step_num}\""

            echo "  Step ${step_num}: ${label}"
            current_section=""
            continue
        fi

        [[ "$in_step" == "0" ]] && continue

        # Parse key-value pairs at step level
        if [[ "$line" =~ ^[[:space:]]+([a-z_]+):[[:space:]]*(.*)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"

            # Clean value (remove quotes, brackets)
            value=$(echo "$value" | sed 's/^["'"'"'\[]//' | sed 's/["'"'"'\]]*$//')

            case "$key" in
                key)
                    eval "STEP_${step_num}_KEY=\"\$value\""
                    ;;
                mirror_hardwares)
                    # Parse array: [amdexperimental, amdproduction]
                    value=$(echo "$value" | tr ',' ' ' | xargs)
                    eval "STEP_${step_num}_MIRROR_HARDWARES=\"\$value\""
                    ;;
                agent_pool)
                    eval "STEP_${step_num}_AGENT_POOL=\"\$value\""
                    ;;
                timeout_in_minutes)
                    eval "STEP_${step_num}_TIMEOUT=\"\$value\""
                    ;;
                fast_check)
                    eval "STEP_${step_num}_FAST_CHECK=\"\$value\""
                    ;;
                working_dir)
                    eval "STEP_${step_num}_WORKING_DIR=\"\$value\""
                    ;;
                queue)
                    # Part of agents section
                    eval "STEP_${step_num}_QUEUE=\"\$value\""
                    ;;
                commands)
                    current_section="commands"
                    eval "STEP_${step_num}_COMMANDS=\"\""
                    list_indent=$indent
                    ;;
                source_file_dependencies)
                    current_section="dependencies"
                    eval "STEP_${step_num}_DEPENDENCIES=\"\""
                    list_indent=$indent
                    ;;
                agents)
                    current_section="agents"
                    ;;
                *)
                    # Check if we left a list section
                    if [[ "$indent" -le "$list_indent" ]] && [[ -n "$current_section" ]]; then
                        current_section=""
                    fi
                    ;;
            esac
        # Parse list items (- item)
        elif [[ "$line" =~ ^[[:space:]]*-[[:space:]]+(.+)$ ]]; then
            local item="${BASH_REMATCH[1]}"
            item=$(echo "$item" | sed 's/^["'"'"']//' | sed 's/["'"'"']$//')

            if [[ "$current_section" == "commands" ]]; then
                local current_cmds
                eval "current_cmds=\"\$STEP_${step_num}_COMMANDS\""
                if [[ -n "$current_cmds" ]]; then
                    eval "STEP_${step_num}_COMMANDS=\"\${current_cmds}||\$item\""
                else
                    eval "STEP_${step_num}_COMMANDS=\"\$item\""
                fi
            elif [[ "$current_section" == "dependencies" ]]; then
                local current_deps
                eval "current_deps=\"\$STEP_${step_num}_DEPENDENCIES\""
                if [[ -n "$current_deps" ]]; then
                    eval "STEP_${step_num}_DEPENDENCIES=\"\${current_deps}||\$item\""
                else
                    eval "STEP_${step_num}_DEPENDENCIES=\"\$item\""
                fi
            fi
        fi
    done < "$yaml_file"

    TOTAL_STEPS=$step_num
}

parse_test_yaml ".buildkite/test-amd.yaml"

echo "Parsed ${TOTAL_STEPS} steps from test-amd.yaml"

# Validate parsing
if [[ "$TOTAL_STEPS" == "0" ]]; then
    echo "Error: No steps found in test-amd.yaml"
    exit 1
fi

echo ""

#==============================================================================
# SECTION 5: TEST FILTERING & SELECTION
#==============================================================================

# Function: Check if step should run based on file dependencies
should_run_step() {
    local step_num="$1"

    # If RUN_ALL_TESTS is set, always run
    if [[ "$RUN_ALL_TESTS" == "1" ]]; then
        return 0
    fi

    # Get step dependencies
    local deps_var="STEP_${step_num}_DEPENDENCIES"
    local dependencies="${!deps_var:-}"

    # If no dependencies specified, always run (catch-all test)
    if [[ -z "$dependencies" ]]; then
        return 0
    fi

    # If no files changed, don't run
    if [[ -z "$CHANGED_FILES" ]] || [[ "$CHANGED_FILE_COUNT" == "0" ]]; then
        return 1
    fi

    # Check if any changed file matches dependencies
    # Dependencies use prefix matching (e.g., "vllm_omni/" matches "vllm_omni/diffusion/model.py")
    local IFS='||'
    local deps_array=($dependencies)

    for dep_pattern in "${deps_array[@]}"; do
        # Remove trailing slashes for consistency
        dep_pattern="${dep_pattern%/}"

        while IFS= read -r changed_file; do
            [[ -z "$changed_file" ]] && continue

            # Check if changed file matches dependency pattern
            # Support both prefix matching and exact file matching
            if [[ "$changed_file" == "${dep_pattern}"* ]] || [[ "$changed_file" == "$dep_pattern" ]]; then
                if [[ "$DEBUG" == "1" ]]; then
                    echo "    Match: ${changed_file} ↔ ${dep_pattern}"
                fi
                return 0
            fi
        done <<< "$CHANGED_FILES"
    done

    return 1
}

# Function: Check if step matches hardware filter
matches_hardware() {
    local step_num="$1"
    local target_hardware="${2:-amdexperimental}"

    # Get mirror_hardwares for this step
    local hw_var="STEP_${step_num}_MIRROR_HARDWARES"
    local hardwares="${!hw_var:-}"

    # If no hardware specified, it's a non-AMD step (CPU/build steps)
    # These should always be included
    if [[ -z "$hardwares" ]]; then
        return 0
    fi

    # Check if target hardware is in the list
    if [[ "$hardwares" == *"$target_hardware"* ]]; then
        return 0
    fi

    return 1
}

echo "--- Filtering tests"
echo "Filter criteria:"
echo "  Target hardware: amdexperimental"
echo "  RUN_ALL_TESTS: ${RUN_ALL_TESTS}"
echo "  Changed files: ${CHANGED_FILE_COUNT}"
echo ""

# Build list of steps to include
INCLUDED_STEPS=()

for ((i=1; i<=TOTAL_STEPS; i++)); do
    label_var="STEP_${i}_LABEL"
    label="${!label_var:-Step $i}"

    # Check hardware match
    if ! matches_hardware "$i" "amdexperimental"; then
        echo "  Step ${i} (${label}): ✗ Wrong hardware - SKIP"
        continue
    fi

    # Check file dependencies
    if should_run_step "$i"; then
        echo "  Step ${i} (${label}): ✓ INCLUDE"
        INCLUDED_STEPS+=("$i")
    else
        echo "  Step ${i} (${label}): ✗ No matching changes - SKIP"
    fi
done

echo ""
echo "Selected ${#INCLUDED_STEPS[@]} of ${TOTAL_STEPS} steps"

# Ensure at least build step (step 1) is included
if [[ "${#INCLUDED_STEPS[@]}" == "0" ]]; then
    echo "Warning: No tests selected, including build step as fallback"
    INCLUDED_STEPS=(1)
fi

echo ""

#==============================================================================
# SECTION 6: PIPELINE GENERATION
#==============================================================================

echo "--- Generating Buildkite pipeline"

# Start pipeline file
cat > .buildkite/pipeline.yaml << 'PIPELINE_HEADER'
# Auto-generated by bootstrap-omni-amd.sh
# DO NOT EDIT MANUALLY - Edit .buildkite/test-amd.yaml instead
#
# Generated:
PIPELINE_HEADER

echo "# $(date -u)" >> .buildkite/pipeline.yaml
echo "" >> .buildkite/pipeline.yaml
echo "steps:" >> .buildkite/pipeline.yaml

# Generate steps
for step_num in "${INCLUDED_STEPS[@]}"; do
    # Get step data
    label_var="STEP_${step_num}_LABEL"
    label="${!label_var}"

    key_var="STEP_${step_num}_KEY"
    key="${!key_var:-step-${step_num}}"

    queue_var="STEP_${step_num}_QUEUE"
    queue="${!queue_var:-}"

    agent_pool_var="STEP_${step_num}_AGENT_POOL"
    agent_pool="${!agent_pool_var:-}"

    timeout_var="STEP_${step_num}_TIMEOUT"
    timeout="${!timeout_var:-10}"

    commands_var="STEP_${step_num}_COMMANDS"
    commands="${!commands_var:-}"

    working_dir_var="STEP_${step_num}_WORKING_DIR"
    working_dir="${!working_dir_var:-}"

    # Determine queue
    final_queue=""
    if [[ -n "$queue" ]]; then
        final_queue="$queue"
    elif [[ -n "$agent_pool" ]]; then
        # Map agent_pool to queue name
        case "$agent_pool" in
            mi325_1) final_queue="amd_mi325_1" ;;
            mi325_2) final_queue="amd_mi325_2" ;;
            mi325_4) final_queue="amd_mi325_4" ;;
            mi325_8) final_queue="amd_mi325_8" ;;
            *) final_queue="amd_${agent_pool}" ;;
        esac
    else
        # Default queue for steps without specification
        final_queue="cpu_queue_premerge"
    fi

    # Generate step YAML
    cat >> .buildkite/pipeline.yaml << STEP_START

  - label: "${label}"
    key: "${key}"
    agents:
      queue: "${final_queue}"
      cluster: "CI"
STEP_START

    # Add commands
    if [[ -n "$commands" ]]; then
        # Check if this is an AMD GPU test (needs run-amd-test.sh wrapper)
        if [[ "$final_queue" == amd_* ]]; then
            # Build command string for AMD GPU execution
            cmd_string=""
            IFS='||'
            cmd_array=($commands)

            # Add ROCm check prefix
            cmd_string="(command rocm-smi || true) && export VLLM_ALLOW_DEPRECATED_BEAM_SEARCH=1"

            # Add working directory if specified
            if [[ -n "$working_dir" ]]; then
                cmd_string="${cmd_string} && cd ${working_dir}"
            fi

            # Join commands with &&
            for cmd in "${cmd_array[@]}"; do
                cmd_string="${cmd_string} && ${cmd}"
            done

            # Wrap in run-amd-test.sh
            echo "    command: bash .buildkite/scripts/hardware_ci/run-amd-test.sh \"${cmd_string}\"" >> .buildkite/pipeline.yaml
        else
            # CPU or build step - direct commands
            echo "    commands:" >> .buildkite/pipeline.yaml
            IFS='||'
            cmd_array=($commands)
            for cmd in "${cmd_array[@]}"; do
                echo "      - \"${cmd}\"" >> .buildkite/pipeline.yaml
            done
        fi
    else
        # No commands specified
        echo "    command: echo \"No commands specified for this step\"" >> .buildkite/pipeline.yaml
    fi

    # Add timeout
    echo "    timeout_in_minutes: ${timeout}" >> .buildkite/pipeline.yaml

    # Add retry for AMD GPU tests
    if [[ "$final_queue" == amd_* ]]; then
        cat >> .buildkite/pipeline.yaml << 'RETRY_BLOCK'
    retry:
      automatic:
        - exit_status: "*"
          limit: 1
RETRY_BLOCK
    fi

    # Add environment variables for AMD GPU tests
    if [[ "$final_queue" == amd_* ]]; then
        cat >> .buildkite/pipeline.yaml << 'ENV_BLOCK'
    env:
      HF_HOME: "/root/.cache/huggingface"
ENV_BLOCK
    fi

    # Add depends_on for non-first steps (depend on build step)
    if [[ "$step_num" != "${INCLUDED_STEPS[0]}" ]]; then
        first_key_var="STEP_${INCLUDED_STEPS[0]}_KEY"
        first_key="${!first_key_var:-step-${INCLUDED_STEPS[0]}}"
        echo "    depends_on: \"${first_key}\"" >> .buildkite/pipeline.yaml
    fi
done

# Validate generated pipeline
if ! grep -q "^steps:" .buildkite/pipeline.yaml; then
    echo "Error: Generated pipeline is invalid (missing 'steps:' section)"
    exit 1
fi

echo "Pipeline generated successfully"
echo ""

# Display generated pipeline
echo "--- Generated Pipeline:"
cat .buildkite/pipeline.yaml
echo ""

# Upload pipeline to Buildkite
echo "--- Uploading pipeline to Buildkite"
buildkite-agent pipeline upload .buildkite/pipeline.yaml

echo ""
echo "=== Bootstrap Complete ==="
echo "Configuration:"
echo "  Total steps defined: ${TOTAL_STEPS}"
echo "  Steps selected: ${#INCLUDED_STEPS[@]}"
echo "  RUN_ALL_TESTS: ${RUN_ALL_TESTS}"
echo "  NO_FAIL_FAST: ${NO_FAIL_FAST}"
echo "  Changed files: ${CHANGED_FILE_COUNT}"
echo ""
echo "Pipeline uploaded successfully!"
