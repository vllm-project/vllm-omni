#!/bin/bash
# Helper function to safely login to ECR Public with per-job config isolation
# Uses DOCKER_CONFIG environment variable to prevent race conditions
#
# This script prevents the "device or resource busy" error by giving each
# Buildkite job its own isolated Docker config directory.
#
# Usage:
#   source docker_login_ecr_public.sh && safe_docker_login_ecr_public

set -euo pipefail

# Configuration
ECR_REGISTRY="public.ecr.aws"

setup_isolated_docker_config() {
    # Use BUILDKITE_JOB_ID for job-specific isolation
    # Fallback to PID if running outside Buildkite
    local job_id="${BUILDKITE_JOB_ID:-$$}"

    # Set Docker config to job-specific directory
    export DOCKER_CONFIG="/tmp/docker-config-${job_id}"

    # Create directory if it doesn't exist
    mkdir -p "$DOCKER_CONFIG"

    echo "[docker-config] Using isolated Docker config: $DOCKER_CONFIG"
}

check_docker_auth() {
    # Check if already authenticated to the given registry
    # Returns 0 if authenticated, 1 if not
    local registry="$1"

    # Check if credentials exist in the isolated config
    if [[ -f "$DOCKER_CONFIG/config.json" ]]; then
        # Check if registry is present in config
        if grep -q "$registry" "$DOCKER_CONFIG/config.json" 2>/dev/null; then
            return 0
        fi
    fi

    return 1
}

safe_docker_login_ecr_public() {
    # Setup isolated config first
    setup_isolated_docker_config

    local registry="$ECR_REGISTRY"

    # Check if already authenticated (within this job)
    if check_docker_auth "$registry"; then
        echo "[docker-login] Already authenticated to $registry in this job"
        return 0
    fi

    # Perform login to isolated config directory
    echo "[docker-login] Logging in to $ECR_REGISTRY (isolated config)..."
    if aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$ECR_REGISTRY"; then
        echo "[docker-login] Login successful (config: $DOCKER_CONFIG)"
        return 0
    else
        local exit_code=$?
        echo "[docker-login] ERROR: Login failed with exit code $exit_code" >&2
        return $exit_code
    fi
}

# Execute if run as script (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    safe_docker_login_ecr_public
fi
