#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Clean up old nightly builds from DockerHub (vllm/vllm-omni), keeping the
# newest 14 tags with the nightly- prefix.

set -ex

TAG_PREFIX="nightly-"
REPO="vllm/vllm-omni"
REPO_API_URL="https://hub.docker.com/v2/repositories/${REPO}/tags"

echo "Cleaning up tags with prefix: $TAG_PREFIX in repository: $REPO"

if [ -z "$DOCKERHUB_TOKEN" ]; then
  echo "Error: DOCKERHUB_TOKEN environment variable is not set"
  exit 1
fi

if [ -z "$DOCKERHUB_USERNAME" ]; then
  echo "Error: DOCKERHUB_USERNAME environment variable is not set"
  exit 1
fi

echo "Getting DockerHub bearer token..."
set +x
BEARER_TOKEN=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$DOCKERHUB_USERNAME\", \"password\": \"$DOCKERHUB_TOKEN\"}" \
  "https://hub.docker.com/v2/users/login" | jq -r '.token')
set -x

if [ -z "$BEARER_TOKEN" ] || [ "$BEARER_TOKEN" = "null" ]; then
  echo "Error: Failed to get DockerHub bearer token"
  exit 1
fi

get_all_tags() {
  local page=1
  local all_tags=""
  # Hub `name` is a partial match (not a strict prefix). Use it to narrow the
  # list, then enforce startswith() client-side. Exhaust API pages; do not
  # stop merely because a page has no matching tags after filtering.
  local name_query
  name_query=$(printf '%s' "$TAG_PREFIX" | jq -sRr @uri)

  while true; do
    set +x
    local response
    response=$(curl -s -H "Authorization: Bearer $BEARER_TOKEN" \
      "$REPO_API_URL?page=$page&page_size=100&name=${name_query}")
    set -x

    local page_count
    page_count=$(echo "$response" | jq -r '.results | length')
    if [ -z "$page_count" ] || [ "$page_count" = "0" ] || [ "$page_count" = "null" ]; then
      break
    fi

    local tags
    tags=$(echo "$response" | jq -r --arg prefix "$TAG_PREFIX" \
      '.results[] | select(.name | startswith($prefix)) | "\(.last_updated)|\(.name)"')

    if [ -n "$tags" ]; then
      all_tags="$all_tags$tags"$'\n'
    fi
    page=$((page + 1))
  done

  echo "$all_tags" | sort -r | cut -d'|' -f2
}

delete_tag() {
  local tag_name="$1"
  echo "Deleting tag: $tag_name"

  local delete_url="https://hub.docker.com/v2/repositories/${REPO}/tags/$tag_name"
  set +x
  local response http_code body
  response=$(curl -s -w "\n%{http_code}" -X DELETE \
    -H "Authorization: Bearer $BEARER_TOKEN" "$delete_url")
  set -x
  http_code=$(printf '%s\n' "$response" | tail -n1)
  body=$(printf '%s\n' "$response" | sed '$d')

  if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
    echo "Successfully deleted tag: $tag_name (HTTP ${http_code})"
    return 0
  fi

  local detail=""
  if echo "$body" | jq -e '.detail' > /dev/null 2>&1; then
    detail=$(echo "$body" | jq -r '.detail')
  elif [ -n "$body" ]; then
    detail="$body"
  else
    detail="(empty response body)"
  fi
  echo "Error: Failed to delete tag $tag_name (HTTP ${http_code}): ${detail}"
  return 1
}

echo "Fetching all tags from DockerHub..."
all_tags=$(get_all_tags)

if [ -z "$all_tags" ]; then
  echo "No tags found to clean up"
  exit 0
fi

total_tags=$(echo "$all_tags" | wc -l)
echo "Found $total_tags tags"

tags_to_keep=14
tags_to_delete=$((total_tags - tags_to_keep))

if [ $tags_to_delete -le 0 ]; then
  echo "No tags need to be deleted (only $total_tags tags found, keeping $tags_to_keep)"
  exit 0
fi

echo "Will delete $tags_to_delete old tags, keeping the newest $tags_to_keep"

tags_to_delete_list=$(echo "$all_tags" | tail -n +$((tags_to_keep + 1)))

if [ -z "$tags_to_delete_list" ]; then
  echo "No tags to delete"
  exit 0
fi

echo "Deleting old tags..."
delete_failures=0
while IFS= read -r tag; do
  if [ -n "$tag" ]; then
    if ! delete_tag "$tag"; then
      delete_failures=$((delete_failures + 1))
    fi
    sleep 1
  fi
done <<< "$tags_to_delete_list"

if [ "$delete_failures" -gt 0 ]; then
  echo "Cleanup finished with ${delete_failures} deletion failure(s)"
  exit 1
fi

echo "Cleanup completed successfully"
