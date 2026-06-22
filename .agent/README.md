# Git Workflow Automation

This directory contains scripts to automate and enforce the Git workflow defined in the [Development Guide](../docs/DEVELOPMENT_GUIDE.md).

## Files

- `git_workflow.sh` - Interactive script to create properly formatted commits
- `pre-commit` - Git hook to enforce commit message format and branch naming

## Setup

1. Make the scripts executable:
   ```bash
   chmod +x .windsurf/git_workflow.sh
   chmod +x .windsurf/pre-commit
   ```

2. Set up the pre-commit hook:
   ```bash
   ln -s ../../.windsurf/pre-commit .git/hooks/pre-commit
   ```

## Usage

### Creating Commits

1. Stage your changes:
   ```bash
   git add <files>
   ```

2. Run the commit script:
   ```bash
   ./.windsurf/git_workflow.sh
   ```

3. Follow the interactive prompts to create a properly formatted commit.

### Branch Naming

Branches must follow this format:
```
<type>/<description>
```

Valid types:
- `feat/` - New features
- `fix/` - Bug fixes
- `hotfix/` - Critical production fixes
- `docs/` - Documentation updates
- `refactor/` - Code improvements without new features
- `chore/` - Maintenance tasks

Examples:
- `feat/user-authentication`
- `fix/login-validation`
- `docs/update-api-docs`

### Commit Message Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

Example:
```
feat(auth): implement login functionality

- Added JWT authentication
- Implemented login/logout endpoints
- Added input validation

Part of MVP-123
```
