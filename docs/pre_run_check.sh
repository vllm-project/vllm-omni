#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# We only run readthedocs when a PR has the trigger label.
TRIGGER_LABEL="ready"

if [ "$READTHEDOCS_VERSION_TYPE" != "external" ]; then
  echo "Not a PR build (version type=$READTHEDOCS_VERSION_TYPE); skipping pre-run-check gate."
  exit 0
fi

# Use a GitHub token if provided to raise the API rate limit (60 -> 5000
# requests/hour). Set GITHUB_TOKEN in the Read the Docs environment variables.
CURL_AUTH=()
if [ -n "$GITHUB_TOKEN" ]; then
  CURL_AUTH=(-H "Authorization: Bearer $GITHUB_TOKEN")
fi


echo "Checking for the '${TRIGGER_LABEL}' label on PR #${READTHEDOCS_VERSION_NAME}..."
LABELS=$(curl -sS "${CURL_AUTH[@]}" "https://api.github.com/repos/vllm-project/vllm-omni/issues/${READTHEDOCS_VERSION_NAME}/labels" | python3 -c "import sys, json; print('\n'.join(l.get('name', '') for l in json.load(sys.stdin)))")
if printf '%s\n' "$LABELS" | grep -qx "${TRIGGER_LABEL}"; then
  echo "PR has the '${TRIGGER_LABEL}' label; forcing build."
  exit 0
fi

echo "PR does not have the '${TRIGGER_LABEL}' label; cancelling build."
# See https://docs.readthedocs.com/platform/latest/guides/build/skip-or-cancel-builds.html for info on exit code
exit 183
