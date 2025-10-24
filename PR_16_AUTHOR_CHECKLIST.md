# PR #16 - Author Checklist

👤 **For**: PR Author (@tzhouam)  
📅 **Updated**: 2025-10-24  
✨ **Goal**: Get PR #16 merged successfully

---

## ✅ Pre-Merge Checklist

### Critical (Must Complete Before Merge)

- [ ] **Fix Variable Name Bugs** (Lines 133-140)
  - [ ] Change `outputs.shape[0]` to `multimodal_outputs.shape[0]` (line 135)
  - [ ] Change `outputs[i]` to `multimodal_outputs[i]` (line 137)
  - [ ] Change `for out in outputs:` to `for out in multimodal_outputs:` (line 139)
  - [ ] Run linter to verify no other similar bugs
  - [ ] Test the fixed code with actual model

- [ ] **Add Test Plan to PR Description**
  - [ ] Describe what will be tested
  - [ ] List test scenarios (single GPU, multi-GPU, etc.)
  - [ ] Document expected outcomes

- [ ] **Run Tests and Document Results**
  - [ ] Unit tests for diffusion runner
  - [ ] Integration tests with actual model
  - [ ] Multi-GPU tests (if applicable)
  - [ ] Add test results to PR description

- [ ] **Clarify KV Transfer Logic** (Lines 47-52)
  - [ ] Decide: Is KV transfer needed for diffusion?
    - [ ] If YES: Add detailed comment explaining why
    - [ ] If NO: Simplify code to just return EMPTY_MODEL_RUNNER_OUTPUT
  - [ ] Update code based on decision

### Important (Should Complete)

- [ ] **Code Cleanup**
  - [ ] Remove or document commented code (lines 191-196)
    - Option A: Remove completely
    - Option B: Convert to TODO comment
  - [ ] Fix excessive blank lines (line 201: reduce from 3 to 1)
  - [ ] Check for unused imports (GPUModelRunner?)
  - [ ] Run code formatter: `black vllm_omni/worker/`

- [ ] **Add Documentation**
  - [ ] Add docstring for `execute_model` method
  - [ ] Add docstring for `_run_diffusion` method
  - [ ] Add docstring for `_dummy_run` method (if override is necessary)
  - [ ] Add class-level docstring explaining when to use this runner

- [ ] **Answer Review Questions**
  - [ ] Is `GPUModelRunner` import used? If not, remove it
  - [ ] Is `_dummy_run` override necessary? If not, remove it
  - [ ] Why is vLLM infrastructure needed vs simple implementation?
  - [ ] Has performance been compared with old implementation?

### Recommended (Nice to Have)

- [ ] **Testing Improvements**
  - [ ] Add unit test for tensor output handling
  - [ ] Add unit test for list output handling
  - [ ] Add unit test for dict output handling
  - [ ] Add unit test for error cases
  - [ ] Add benchmark comparing old vs new implementation

- [ ] **Documentation**
  - [ ] Add architecture diagram showing data flow
  - [ ] Document design decisions in PR description
  - [ ] Add usage examples
  - [ ] Update any affected documentation files

- [ ] **Code Quality**
  - [ ] Run type checker: `mypy vllm_omni/worker/`
  - [ ] Run linter: `ruff check vllm_omni/worker/`
  - [ ] Review for any TODO comments that should be addressed
  - [ ] Check for proper error handling

---

## 📋 Step-by-Step Guide

### Step 1: Apply Bug Fixes (30 mins)

1. Open `vllm_omni/worker/gpu_diffusion_model_runner.py`
2. Go to lines 133-140
3. Replace all instances of `outputs` with `multimodal_outputs` in that block
4. Save and verify with: `git diff vllm_omni/worker/gpu_diffusion_model_runner.py`

**See detailed fixes in**: `PR_16_FIX_GUIDE.md`

### Step 2: Code Cleanup (15 mins)

1. Remove commented code (lines 191-196) or convert to TODO
2. Fix blank lines (line 201)
3. Check imports (line 13 - GPUModelRunner)
4. Run formatter: `black vllm_omni/worker/`

### Step 3: Add Tests (2-3 hours)

1. Create test file if doesn't exist: `tests/worker/test_diffusion_model_runner.py`
2. Add unit tests for each output type (tensor, list, dict)
3. Add integration test with mock model
4. Run tests: `pytest tests/worker/ -v`
5. Document results

### Step 4: Update PR Description (15 mins)

1. Add test plan section:
   ```markdown
   ## Test Plan
   - Unit tests for output handling (tensor, list, dict)
   - Integration test with Qwen 2.5 Omni model
   - Multi-GPU test on 2x A100 GPUs
   ```

2. Add test results section:
   ```markdown
   ## Test Results
   - All unit tests pass ✓
   - Integration test passes ✓
   - Multi-GPU test passes ✓
   - Performance: [add numbers]
   ```

### Step 5: Respond to Review Comments (30 mins)

1. Address each review comment
2. Respond with either:
   - "Fixed in [commit hash]"
   - "Resolved by [explanation]"
   - "Will address in follow-up PR because [reason]"

### Step 6: Re-Request Review (5 mins)

1. Verify all critical items are checked
2. Re-request review from reviewers
3. Add comment summarizing what was changed

---

## 🔍 Self-Review Checklist

Before re-requesting review, verify:

- [ ] All code compiles without errors
- [ ] All tests pass
- [ ] No lint errors
- [ ] Git diff shows only intended changes
- [ ] PR description is complete and accurate
- [ ] All review comments have responses
- [ ] No debugging code or print statements left behind
- [ ] No sensitive information (tokens, passwords) in code

---

## 📊 Progress Tracker

| Category | Total Items | Completed | Percentage |
|----------|-------------|-----------|------------|
| Critical | 4 | 0 | 0% |
| Important | 4 | 0 | 0% |
| Recommended | 4 | 0 | 0% |
| **Overall** | **12** | **0** | **0%** |

*Update this as you complete items*

---

## 🎯 Quick Wins (Do These First)

1. ✅ Fix variable name bugs (15 mins, huge impact)
2. ✅ Fix blank lines (1 min, easy)
3. ✅ Remove commented code (5 mins, clean)
4. ✅ Add basic test (1 hour, shows due diligence)
5. ✅ Update PR description (15 mins, looks professional)

**Total**: ~2 hours to address most critical issues!

---

## ❓ Common Questions

### Q: Do I need to fix everything before re-requesting review?
**A**: No, but you MUST fix:
- Variable name bugs
- Add test plan/results
- Clean up commented code

Other items can be follow-up PRs if agreed with reviewers.

### Q: How do I know if KV transfer is needed?
**A**: Ask yourself:
- Does the diffusion model use encoder outputs from another stage?
- Do you have pipeline parallelism across multiple GPUs?
- If NO to both: probably don't need it

### Q: Can I merge with TODO comments?
**A**: Yes, if:
- TODOs are documented properly
- Not for critical functionality
- Have agreement from reviewers

### Q: What if I can't test on multi-GPU?
**A**: Document this limitation:
- Note in PR description: "Tested on single GPU only"
- Ask reviewers with multi-GPU access to test
- Add TODO for multi-GPU testing

---

## 🆘 Need Help?

### Stuck on bug fixes?
→ See `PR_16_FIX_GUIDE.md` for copy-paste solutions

### Need more context?
→ See `PR_16_EXPERT_REVIEW.md` for detailed analysis

### Want to understand business impact?
→ See `PR_16_EXECUTIVE_SUMMARY.md` for stakeholder view

### Quick reference?
→ See `PR_16_REVIEW_SUMMARY.md` for key points

---

## 🚀 When You're Done

1. [ ] All critical items checked ✓
2. [ ] Tests pass ✓
3. [ ] PR description updated ✓
4. [ ] Review comments addressed ✓
5. [ ] **Re-request review from**:
   - [ ] @hsliuustc0106
   - [ ] @fake0fan
   - [ ] @Gaohan123
   - [ ] @congw729

---

## 🎉 Success Criteria

Your PR is ready when:
- ✅ No critical bugs remain
- ✅ Tests pass and results documented
- ✅ Code is clean and documented
- ✅ All reviewer questions answered
- ✅ Reviewers approve

**Then**: PR gets merged! 🎊

---

**Good luck! You've got this! 💪**

*Remember: It's better to take time to do it right than to rush and need multiple fix-up commits.*

---

**Last Updated**: 2025-10-24  
**Next Review**: After author completes critical items
