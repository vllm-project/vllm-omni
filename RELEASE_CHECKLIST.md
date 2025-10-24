# Release Checklist

Use this checklist when preparing a new release of vLLM-omni.

## Pre-Release (1-2 weeks before)

### Code Quality
- [ ] All tests pass on main branch
- [ ] Code coverage is adequate (>70% ideally)
- [ ] No critical or high severity security issues
- [ ] All linting checks pass (black, isort, flake8, mypy)
- [ ] Documentation is up to date

### Version Management
- [ ] Decide on version number (following semantic versioning)
- [ ] Update version in `pyproject.toml`
- [ ] Update version in `vllm_omni/__init__.py`
- [ ] Update version in documentation if referenced

### Changelog
- [ ] Review all merged PRs since last release
- [ ] Update CHANGELOG.md with new features, bug fixes, breaking changes
- [ ] Categorize changes appropriately (Added, Changed, Deprecated, Removed, Fixed, Security)
- [ ] Add release date to CHANGELOG.md
- [ ] Ensure all major changes are documented

### Documentation
- [ ] Review and update README.md
- [ ] Update installation instructions if needed
- [ ] Verify all example code works with new version
- [ ] Update API documentation
- [ ] Check all links in documentation are valid
- [ ] Update screenshots/images if UI changed

### Dependencies
- [ ] Review and update dependencies in requirements.txt
- [ ] Review and update dependencies in pyproject.toml
- [ ] Check for security vulnerabilities in dependencies
- [ ] Verify compatibility with latest vLLM version
- [ ] Test with supported Python versions

### Testing
- [ ] Run full test suite on main branch
- [ ] Test installation from source
- [ ] Test examples and tutorials
- [ ] Test on different Python versions (3.8, 3.9, 3.10, 3.11, 3.12)
- [ ] Test on different platforms (Linux, macOS if possible)
- [ ] Perform manual testing of key features

## Release Day

### Final Checks
- [ ] Ensure main branch is stable
- [ ] All CI/CD checks pass
- [ ] No blocking issues in issue tracker
- [ ] Review open security issues

### Version Tagging
- [ ] Create release branch (e.g., `release/v0.1.0`)
- [ ] Final version bump if needed
- [ ] Commit all version changes
- [ ] Push release branch

### Build and Test
- [ ] Build distribution packages locally
  ```bash
  python -m build
  ```
- [ ] Verify package contents
  ```bash
  tar -tzf dist/vllm-omni-*.tar.gz
  ```
- [ ] Test installation from built package
  ```bash
  pip install dist/vllm-omni-*.whl
  ```
- [ ] Run smoke tests with installed package

### Git Operations
- [ ] Merge release branch to main
- [ ] Create and push git tag
  ```bash
  git tag -a v0.1.0 -m "Release version 0.1.0"
  git push origin v0.1.0
  ```

### GitHub Release
- [ ] Go to GitHub Releases page
- [ ] Create new release from tag
- [ ] Use semantic version (e.g., v0.1.0)
- [ ] Copy relevant section from CHANGELOG.md
- [ ] Add release notes highlighting major changes
- [ ] Attach built distribution files (optional)
- [ ] Mark as pre-release if appropriate
- [ ] Publish release

### PyPI Publication
- [ ] Verify PyPI credentials are configured
- [ ] Upload to Test PyPI first
  ```bash
  twine upload --repository testpypi dist/*
  ```
- [ ] Test installation from Test PyPI
  ```bash
  pip install --index-url https://test.pypi.org/simple/ vllm-omni
  ```
- [ ] Upload to production PyPI
  ```bash
  twine upload dist/*
  ```
  Or let GitHub Actions handle it automatically
- [ ] Verify package appears on PyPI
- [ ] Test installation from PyPI
  ```bash
  pip install vllm-omni
  ```

### Communication
- [ ] Announce release in GitHub Discussions
- [ ] Update project website if applicable
- [ ] Share on social media if appropriate
- [ ] Notify key stakeholders
- [ ] Update any related documentation sites

## Post-Release

### Immediate Follow-up
- [ ] Monitor issue tracker for new issues
- [ ] Monitor PyPI download statistics
- [ ] Check CI/CD for any failures
- [ ] Respond to release-related questions

### Version Management
- [ ] Create next development version
- [ ] Update CHANGELOG.md with [Unreleased] section
- [ ] Commit and push version updates

### Documentation
- [ ] Update ReadTheDocs or documentation site
- [ ] Archive old version documentation if needed
- [ ] Update compatibility matrices

### Housekeeping
- [ ] Close milestone for this release
- [ ] Review and close completed issues
- [ ] Update roadmap/project board
- [ ] Plan next release features

## Rollback Plan (if issues found)

If critical issues are discovered immediately after release:

1. **Assess Severity**
   - [ ] Determine if issue warrants immediate rollback
   - [ ] Document the issue clearly

2. **Quick Fix (if possible)**
   - [ ] Prepare hotfix
   - [ ] Fast-track testing
   - [ ] Release patch version (e.g., 0.1.1)

3. **Rollback (if necessary)**
   - [ ] Yank release from PyPI (doesn't delete, marks as unavailable)
   - [ ] Update GitHub release notes with warning
   - [ ] Announce rollback to users
   - [ ] Investigate and fix root cause

## Version Numbering Guide

Follow Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR** (e.g., 1.0.0): Breaking changes, incompatible API changes
- **MINOR** (e.g., 0.2.0): New features, backwards compatible
- **PATCH** (e.g., 0.1.1): Bug fixes, backwards compatible

Pre-release versions:
- **Alpha** (e.g., 0.1.0a1): Early testing, unstable
- **Beta** (e.g., 0.1.0b1): Feature complete, testing phase
- **RC** (e.g., 0.1.0rc1): Release candidate, final testing

## Notes

- Keep this checklist updated as release process evolves
- Document any issues encountered during release
- Learn from each release to improve the process
- Automate repetitive tasks when possible

---

**Release Manager**: _[Name]_
**Release Date**: _[Date]_
**Version**: _[Version]_
