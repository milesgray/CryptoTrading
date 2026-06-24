#!/bin/bash
set -e

PR_NUMBER=$1
REVIEW_FILE=$2

if [ -z "$PR_NUMBER" ] || [ -z "$REVIEW_FILE" ]; then
  echo "Usage: $0 <pr-number> <review-file>"
  exit 1
fi

if [ ! -f "$REVIEW_FILE" ]; then
  echo "Error: Review file '$REVIEW_FILE' not found."
  exit 1
fi

# Get repository details from git
REMOTE_URL=$(git remote get-url origin)
# Extract owner and repo from URL (works for both HTTPS and SSH formats)
if [[ "$REMOTE_URL" =~ github\.com[:/]([^/]+)/([^.]+)(\.git)? ]]; then
  OWNER="${BASH_REMATCH[1]}"
  REPO="${BASH_REMATCH[2]}"
else
  echo "Error: Could not parse GitHub owner and repo from remote URL: $REMOTE_URL"
  exit 1
fi

echo "Posting review to $OWNER/$REPO PR #$PR_NUMBER..."

# Extract summary and comments from the JSON review file
SUMMARY=$(jq -r '.summary' "$REVIEW_FILE")
COMMENTS=$(jq '.comments' "$REVIEW_FILE")

# Construct the payload for the GitHub PR Review API
# The API accepts: { body, event: "COMMENT", comments: [ { path, line, side, body } ] }
PAYLOAD=$(jq -n \
  --arg body "$SUMMARY" \
  --argjson comments "$COMMENTS" \
  '{body: $body, event: "COMMENT", comments: $comments}')

# Post the review using the gh CLI API
gh api \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  --method POST \
  "/repos/$OWNER/$REPO/pulls/$PR_NUMBER/reviews" \
  --input - <<< "$PAYLOAD"

echo "Review posted successfully!"
