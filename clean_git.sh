#!/bin/bash
# Script to remove all .git files from git tracking

echo "Starting cleanup of .git files from git index..."

# Get all .git files currently tracked
git_files=$(git ls-files | grep "\.git")

if [ -z "$git_files" ]; then
    echo "No .git files found in git index"
    exit 0
fi

echo "Found $(echo "$git_files" | wc -l) .git files to remove"

# Remove each file
echo "$git_files" | while read -r file; do
    if [ -f "$file" ]; then
        git rm --cached "$file" 2>/dev/null
        echo "Removed: $file"
    fi
done

echo "Cleanup complete. Please commit the changes."
