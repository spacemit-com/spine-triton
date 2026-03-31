#!/usr/bin/env bash
set -euo pipefail

# create_pr.sh
# Usage:
#   GITHUB_TOKEN=<token> ./scripts/create_pr.sh [-r remote] [-b branch] [-o target_owner] [-R target_repo] [-B base_branch] -t "Title" -d "Body"
# Defaults:
#   remote=origin, branch=current, target_owner=spacemit-com, target_repo=spine-triton, base_branch=main

REMOTE=origin
BRANCH=""
TARGET_OWNER=spacemit-com
TARGET_REPO=spine-triton
BASE_BRANCH=main
TITLE=""
BODY=""

usage() {
  cat <<EOF
Usage: GITHUB_TOKEN=<token> $0 [-r remote] [-b branch] [-o target_owner] [-R target_repo] [-B base_branch] -t "Title" -d "Body"
EOF
  exit 1
}

while getopts ":r:b:o:R:B:t:d:h" opt; do
  case ${opt} in
    r ) REMOTE=$OPTARG ;;
    b ) BRANCH=$OPTARG ;;
    o ) TARGET_OWNER=$OPTARG ;;
    R ) TARGET_REPO=$OPTARG ;;
    B ) BASE_BRANCH=$OPTARG ;;
    t ) TITLE=$OPTARG ;;
    d ) BODY=$OPTARG ;;
    h ) usage ;;
    \? ) usage ;;
  esac
done

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Error: GITHUB_TOKEN must be set (Personal Access Token with repo scope)" >&2
  exit 2
fi

if [[ -z "$BRANCH" ]]; then
  BRANCH=$(git rev-parse --abbrev-ref HEAD)
fi

if [[ -z "$TITLE" ]]; then
  TITLE="Auto PR: ${BRANCH}"
fi

# get remote url and derive owner
REMOTE_URL=$(git remote get-url "$REMOTE" 2>/dev/null || true)
if [[ -z "$REMOTE_URL" ]]; then
  echo "Remote '$REMOTE' not found" >&2
  exit 3
fi

parse_owner_from_url() {
  local url="$1"
  # git@github.com:owner/repo.git
  if [[ "$url" =~ ^git@github.com:([^/]+)/([^/.]+)(\.git)?$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  # https://github.com/owner/repo.git or https://github.com/owner/repo
  if [[ "$url" =~ ^https?://github.com/([^/]+)/([^/.]+)(\.git)? ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  # fallback: try splitting
  echo "$url" | awk -F'[/:]' '{print $(NF-1)}'
}

FORK_OWNER=$(parse_owner_from_url "$REMOTE_URL")
if [[ -z "$FORK_OWNER" ]]; then
  echo "Failed to parse owner from remote URL: $REMOTE_URL" >&2
  exit 4
fi

echo "Pushing branch '$BRANCH' to remote '$REMOTE' (owner: $FORK_OWNER)"
git push "$REMOTE" "$BRANCH"

# create PR via GitHub API
API_URL="https://api.github.com/repos/${TARGET_OWNER}/${TARGET_REPO}/pulls"

read -r -d '' PAYLOAD <<EOF || true
{
  "title": $(jq -Rs <<< "$TITLE"),
  "head": $(jq -Rs <<< "${FORK_OWNER}:${BRANCH}"),
  "base": $(jq -Rs <<< "$BASE_BRANCH"),
  "body": $(jq -Rs <<< "$BODY")
}
EOF

if ! command -v jq >/dev/null 2>&1; then
  # Minimal JSON escaping fallback
  esc() { printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'; }
  PAYLOAD="{\"title\":$(esc "$TITLE"),\"head\":$(esc "${FORK_OWNER}:${BRANCH}"),\"base\":$(esc "$BASE_BRANCH"),\"body\":$(esc "$BODY")}"
fi

echo "Creating PR against ${TARGET_OWNER}/${TARGET_REPO} (${BASE_BRANCH})..."
resp=$(curl -s -S -X POST -H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json" "$API_URL" -d "$PAYLOAD")

pr_url=$(echo "$resp" | grep -o '"html_url": *"[^"]*"' | head -n1 | sed -E 's/"html_url": *"([^"]*)"/\1/') || pr_url=""
if [[ -n "$pr_url" ]]; then
  echo "PR created: $pr_url"
  exit 0
fi

# Try to print message from API
echo "Failed to create PR. Response:"
echo "$resp"
exit 5
