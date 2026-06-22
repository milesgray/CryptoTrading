#!/bin/bash

# Git Workflow Automation Script
# This script helps maintain a clean git history with standardized commits

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to get current branch name
get_current_branch() {
    git rev-parse --abbrev-ref HEAD
}

# Function to check if working directory is clean
is_clean_working_dir() {
    if [ -z "$(git status --porcelain)" ]; then
        return 0
    else
        return 1
    fi
}

# Function to create a properly formatted commit
create_commit() {
    local type=$1
    local scope=$2
    local message=$3
    local body=$4
    local issue=$5

    # Build commit message
    local commit_msg="$type($scope): $message\n\n$body"

    # Add issue reference if provided
    if [ -n "$issue" ]; then
        commit_msg="$commit_msg\n\nPart of $issue"
    fi

    # Create commit
    git commit -m "$commit_msg"
}

# Function to check branch naming convention
validate_branch_name() {
    local branch_name=$(get_current_branch)

    # Skip validation for main/master branches
    if [[ "$branch_name" == "main" || "$branch_name" == "master" ]]; then
        return 0
    fi

    # Check branch name against pattern
    if ! [[ "$branch_name" =~ ^(feat|fix|hotfix|docs|refactor|chore)/[a-z0-9-]+$ ]]; then
        echo -e "${RED}Error: Branch name '$branch_name' does not follow naming convention.${NC}"
        echo -e "Please use one of these formats:"
        echo -e "- feat/feature-name"
        echo -e "- fix/bug-description"
        echo -e "- docs/update-readme"
        echo -e "- refactor/component-name"
        echo -e "- chore/task-description"
        return 1
    fi

    return 0
}

# Main function
main() {
    # Check if we're in a git repository
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo -e "${RED}Error: Not a git repository${NC}"
        return 1
    fi

    # Validate branch name
    if ! validate_branch_name; then
        return 1
    fi

    # Check for staged changes
    if [ -z "$(git diff --cached --name-only)" ]; then
        echo -e "${YELLOW}No staged changes to commit. Add changes with 'git add'.${NC}"
        return 0
    fi

    # Get commit details
    echo -e "${GREEN}Preparing commit...${NC}"

    # Determine commit type from branch name
    local branch_name=$(get_current_branch)
    local default_type="${branch_name%%/*}"

    # Map branch prefix to commit type
    case "$default_type" in
        "feat") default_type="feat" ;;
        "fix") default_type="fix" ;;
        "hotfix") default_type="fix" ;;
        "docs") default_type="docs" ;;
        "refactor") default_type="refactor" ;;
        *) default_type="chore" ;;
    esac

    # Get commit details
    read -p "Commit type [$default_type]: " type
    type=${type:-$default_type}

    read -p "Scope (e.g., component name): " scope
    read -p "Short description: " message
    read -p "Detailed description (optional):\n" body
    read -p "Issue/ticket number (e.g., MVP-123, leave empty if none): " issue

    # Create commit
    create_commit "$type" "$scope" "$message" "$body" "$issue"

    echo -e "${GREEN}✓ Commit created successfully!${NC}"
}

# Run the script
main "$@"
