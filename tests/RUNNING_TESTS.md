# Running Tests - Complete Guide

This guide shows how to run tests at every level, from individual tests to the full suite.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Running by Scope](#running-by-scope)
3. [Running by Test File](#running-by-test-file)
4. [Running by Test Class](#running-by-test-class)
5. [Running Individual Tests](#running-individual-tests)
6. [Running by Keyword](#running-by-keyword)
7. [Running with Options](#running-with-options)
8. [Common Workflows](#common-workflows)

---

## 🚀 Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run all tests with minimal output
pytest tests/ -q

# Run and stop on first failure
pytest tests/ -x

# Run with live output
pytest tests/ -v -s
```

---

## 🎯 Running by Scope

### All Tests (Full Suite - 176 tests)

```bash
# Verbose output
pytest tests/ -v

# Show test names and results
pytest tests/ -v --tb=short

# Quiet mode (just dots)
pytest tests/

# Very verbose (shows more details)
pytest tests/ -vv
```

**Expected output:**
```
========================== test session starts ==========================
collected 176 items

tests/test_document_parser.py::TestCustomerOrderValidation::test_valid_order_creation PASSED
tests/test_document_parser.py::TestCustomerOrderValidation::test_order_with_optional_fields_none PASSED
...
========================== 176 passed in 120s ===========================
```

### All Tests in Specific Directory

```bash
# Run only tests in tests/ directory
pytest tests/ -v

# Run tests recursively
pytest tests/ --recursive -v
```

---

## 📁 Running by Test File

### Document Parser Tests (25 tests)

```bash
# Run all document parser tests
pytest tests/test_document_parser.py -v

# With summary at end
pytest tests/test_document_parser.py -v --tb=short

# Show print statements
pytest tests/test_document_parser.py -v -s
```

**Example output:**
```
========================== test session starts ==========================
collected 25 items

tests/test_document_parser.py::TestCustomerOrderValidation::test_valid_order_creation PASSED [ 4%]
tests/test_document_parser.py::TestCustomerOrderValidation::test_order_with_optional_fields_none PASSED [ 8%]
...
========================== 25 passed in 25s ============================
```

### Text Classifier Tests (33 tests)

```bash
pytest tests/test_text_classifier.py -v
```

### React Agent Tests (25 tests)

```bash
pytest tests/test_react_agent.py -v
```

### RAG QA System Tests (30 tests)

```bash
pytest tests/test_rag_qa_system.py -v
```

### Multi-Step Agent Tests (28 tests)

```bash
pytest tests/test_multi_step_agent.py -v
```

### Application Pipeline Tests (35 tests)

```bash
pytest tests/test_application_pipeline.py -v
```

---

## 🏗️ Running by Test Class

### Syntax

```bash
pytest <file>::<TestClass> -v
```

### Document Parser - Validation Tests Only

```bash
# Run just the validation test class (11 tests)
pytest tests/test_document_parser.py::TestCustomerOrderValidation -v
```

**Example output:**
```
collected 11 items

tests/test_document_parser.py::TestCustomerOrderValidation::test_valid_order_creation PASSED
tests/test_document_parser.py::TestCustomerOrderValidation::test_order_with_optional_fields_none PASSED
tests/test_document_parser.py::TestCustomerOrderValidation::test_negative_price_validation PASSED
...
========================== 11 passed in 5s =============================
```

### Text Classifier - Classification Tests Only

```bash
# Run just InquiryClassifier tests
pytest tests/test_text_classifier.py::TestInquiryClassifier -v
```

### React Agent - Tool Tests Only

```bash
# Run just the tool tests (8 tests, no API calls!)
pytest tests/test_react_agent.py::TestTools -v
```

**Output:**
```
collected 8 items

tests/test_react_agent.py::TestTools::test_calculator_addition PASSED
tests/test_react_agent.py::TestTools::test_calculator_multiplication PASSED
tests/test_react_agent.py::TestTools::test_calculator_complex_expression PASSED
...
========================== 8 passed in 1.23s ============================
```

### RAG System - Question Answering Tests Only

```bash
pytest tests/test_rag_qa_system.py::TestQuestionAnswering -v
```

### Multi-Step Agent - Decision Tests Only

```bash
pytest tests/test_multi_step_agent.py::TestFulfillmentDecision -v
```

### Application Pipeline - End-to-End Tests Only

```bash
pytest tests/test_application_pipeline.py::TestEndToEnd -v
```

---

## 🎯 Running Individual Tests

### Syntax

```bash
pytest <file>::<TestClass>::<test_method> -v
```

### Examples

#### Single Validation Test

```bash
pytest tests/test_document_parser.py::TestCustomerOrderValidation::test_valid_order_creation -v
```

**Output:**
```
collected 1 item

tests/test_document_parser.py::TestCustomerOrderValidation::test_valid_order_creation PASSED [100%]

========================== 1 passed in 2s ===============================
```

#### Single Classification Test

```bash
pytest tests/test_text_classifier.py::TestInquiryClassifier::test_classify_product_question -v
```

#### Single Tool Test

```bash
pytest tests/test_react_agent.py::TestTools::test_calculator_addition -v
```

#### Single RAG Test

```bash
pytest tests/test_rag_qa_system.py::TestQuestionAnswering::test_ask_about_shipping -v
```

#### Single Agent Test

```bash
pytest tests/test_react_agent.py::TestAgent::test_agent_calculator_query -v
```

#### Parametrized Test - Specific Case

```bash
# Run specific parametrized test case
pytest tests/test_document_parser.py::TestParametrizedScenarios::test_quantity_validation_scenarios[1-True] -v
```

---

## 🔍 Running by Keyword

### Match Test Names

```bash
# Run all tests with "validation" in the name
pytest tests/ -k "validation" -v

# Run all tests with "extract" in the name
pytest tests/ -k "extract" -v

# Run all tests with "error" in the name
pytest tests/ -k "error" -v
```

**Example output:**
```
========================== test session starts ==========================
collected 176 items / 150 deselected / 26 selected

tests/test_document_parser.py::TestCustomerOrderValidation::test_negative_price_validation PASSED
tests/test_document_parser.py::TestCustomerOrderValidation::test_zero_price_validation PASSED
...
========================== 26 passed, 150 deselected in 30s ==============
```

### Combine Keywords

```bash
# Run tests matching "classification" OR "routing"
pytest tests/ -k "classification or routing" -v

# Run tests matching "agent" AND "tool"
pytest tests/ -k "agent and tool" -v

# Run tests NOT matching "slow"
pytest tests/ -k "not slow" -v
```

### Keyword Examples by Feature

```bash
# All extraction tests
pytest tests/ -k "extract" -v

# All validation tests
pytest tests/ -k "validation" -v

# All error handling tests
pytest tests/ -k "error" -v

# All edge case tests
pytest tests/ -k "edge" -v

# All initialization tests
pytest tests/ -k "initialization" -v

# All decision tests
pytest tests/ -k "decision" -v
```

---

## ⚙️ Running with Options

### Output Control

```bash
# Verbose output
pytest tests/ -v

# Very verbose (shows more details)
pytest tests/ -vv

# Quiet mode (minimal output)
pytest tests/ -q

# Show print statements
pytest tests/ -v -s

# No header/summary
pytest tests/ --no-header --no-summary
```

### Stop on Failures

```bash
# Stop after first failure
pytest tests/ -x

# Stop after 3 failures
pytest tests/ --maxfail=3

# Continue on failures (default)
pytest tests/
```

### Traceback Control

```bash
# Short traceback
pytest tests/ -v --tb=short

# Long traceback (default)
pytest tests/ -v --tb=long

# No traceback
pytest tests/ -v --tb=no

# Line-only traceback
pytest tests/ -v --tb=line
```

### Show Durations

```bash
# Show 5 slowest tests
pytest tests/ -v --durations=5

# Show 10 slowest tests
pytest tests/ -v --durations=10

# Show all test durations
pytest tests/ -v --durations=0
```

**Example output:**
```
========================== slowest 5 durations ===========================
15.23s call     tests/test_text_classifier.py::test_classify_urgent_order_issue
8.45s call      tests/test_rag_qa_system.py::test_ask_about_shipping
7.89s call      tests/test_multi_step_agent.py::test_process_complete_order
...
```

### Test Collection

```bash
# Show what tests would run (don't execute)
pytest tests/ --collect-only

# Show with short test IDs
pytest tests/ --collect-only -q

# Count tests
pytest tests/ --collect-only -q | tail -n 1
```

**Example output:**
```
collected 176 items

<Dir tests>
  <Module test_document_parser.py>
    <Class TestCustomerOrderValidation>
      <Function test_valid_order_creation>
      <Function test_order_with_optional_fields_none>
...
176 tests collected
```

### Coverage Reports

```bash
# Run with coverage
pytest tests/ --cov=src --cov-report=term

# Coverage with missing lines
pytest tests/ --cov=src --cov-report=term-missing

# HTML coverage report
pytest tests/ --cov=src --cov-report=html
# Then open: htmlcov/index.html

# Multiple report formats
pytest tests/ --cov=src --cov-report=term --cov-report=html
```

### Parallel Execution

```bash
# Install first: pip install pytest-xdist

# Run with auto CPU detection
pytest tests/ -n auto

# Run with specific number of workers
pytest tests/ -n 4

# Run tests in parallel, show output
pytest tests/ -n auto -v
```

---

## 💼 Common Workflows

### 1. Quick Validation (No API Calls)

Run just the validation tests that don't require API calls:

```bash
# Document parser validation only (fast!)
pytest tests/test_document_parser.py::TestCustomerOrderValidation -v

# React agent tools only (no API)
pytest tests/test_react_agent.py::TestTools -v

# Combined
pytest tests/test_document_parser.py::TestCustomerOrderValidation tests/test_react_agent.py::TestTools -v
```

**Time:** ~5-10 seconds

### 2. Smoke Test (One Test Per File)

Quick sanity check:

```bash
pytest tests/test_document_parser.py::TestCustomerOrderValidation::test_valid_order_creation \
       tests/test_text_classifier.py::TestInquiryClassifier::test_classifier_initialization \
       tests/test_react_agent.py::TestTools::test_calculator_addition \
       tests/test_rag_qa_system.py::TestInitialization::test_basic_rag_creation \
       tests/test_multi_step_agent.py::TestInitialization::test_agent_creation \
       tests/test_application_pipeline.py::TestInitialization::test_processor_creation \
       -v
```

**Time:** ~30 seconds

### 3. Feature-Specific Testing

```bash
# Test all extraction functionality
pytest tests/ -k "extract" -v

# Test all validation functionality
pytest tests/ -k "validation" -v

# Test all routing functionality
pytest tests/ -k "routing" -v

# Test all RAG functionality
pytest tests/test_rag_qa_system.py -v
```

### 4. Pre-Commit Testing

Run before committing code:

```bash
# Run fast tests first
pytest tests/test_react_agent.py::TestTools -v

# If those pass, run everything
pytest tests/ -v --tb=short

# Or use maxfail to stop early
pytest tests/ -v --maxfail=1
```

### 5. CI/CD Pipeline Testing

```bash
# Run with coverage and stop on first failure
pytest tests/ -v --cov=src --cov-report=xml --maxfail=1

# Run in parallel with coverage
pytest tests/ -n auto --cov=src --cov-report=term-missing

# Generate multiple reports
pytest tests/ -v --cov=src --cov-report=term --cov-report=html --cov-report=xml
```

### 6. Debugging Specific Test

```bash
# Run with full output
pytest tests/test_document_parser.py::TestOrderExtractor::test_extract_simple_order -v -s

# Run with Python debugger on failure
pytest tests/test_document_parser.py::TestOrderExtractor::test_extract_simple_order -v --pdb

# Run with verbose traceback
pytest tests/test_document_parser.py::TestOrderExtractor::test_extract_simple_order -vv --tb=long
```

### 7. Testing After Changes

```bash
# Run tests related to what you changed
pytest tests/test_document_parser.py -v  # If you changed document_parser.py

# Run with coverage to see what's tested
pytest tests/test_document_parser.py --cov=src.document_parser --cov-report=term-missing
```

### 8. Regression Testing

```bash
# Run all tests, continue on failures, show summary
pytest tests/ -v --tb=short --maxfail=999

# Save results to file
pytest tests/ -v > test_results.txt 2>&1

# Run and generate HTML report
pytest tests/ -v --html=report.html --self-contained-html
```

---

## 📊 Understanding Test Output

### Successful Test Output

```
tests/test_document_parser.py::TestCustomerOrderValidation::test_valid_order_creation PASSED [100%]
```

- `PASSED` - Test succeeded ✅
- `[100%]` - Progress indicator

### Failed Test Output

```
tests/test_document_parser.py::TestOrderExtractor::test_extract_simple_order FAILED [50%]

================================ FAILURES ================================
____ TestOrderExtractor.test_extract_simple_order ____

    def test_extract_simple_order(self, extractor, simple_order):
        result = extractor.extract(simple_order)
>       assert result.customer_name == "Expected Name"
E       AssertionError: assert 'Actual Name' == 'Expected Name'

tests/test_document_parser.py:123: AssertionError
```

### Skipped Test Output

```
tests/test_rag_qa_system.py::TestQuestionAnswering::test_ask_about_shipping SKIPPED [25%]
```

- `SKIPPED` - Test was skipped (e.g., no API key)

---

## 🔧 Useful Flags Reference

| Flag | Purpose | Example |
|------|---------|---------|
| `-v` | Verbose output | `pytest tests/ -v` |
| `-vv` | Very verbose | `pytest tests/ -vv` |
| `-q` | Quiet mode | `pytest tests/ -q` |
| `-s` | Show print statements | `pytest tests/ -s` |
| `-x` | Stop on first failure | `pytest tests/ -x` |
| `--maxfail=N` | Stop after N failures | `pytest tests/ --maxfail=3` |
| `-k` | Run tests matching keyword | `pytest tests/ -k "validation"` |
| `--tb=short` | Short traceback | `pytest tests/ --tb=short` |
| `--durations=N` | Show N slowest tests | `pytest tests/ --durations=5` |
| `--collect-only` | Show tests without running | `pytest tests/ --collect-only` |
| `--cov=src` | Coverage for src/ | `pytest tests/ --cov=src` |
| `-n auto` | Run in parallel | `pytest tests/ -n auto` |
| `--pdb` | Drop into debugger on failure | `pytest tests/ --pdb` |

---

## 💡 Tips & Tricks

### 1. Chain Multiple Filters

```bash
# Run validation tests in document_parser only
pytest tests/test_document_parser.py -k "validation" -v

# Run extraction tests, stop on first failure
pytest tests/ -k "extract" -x -v

# Run edge case tests with coverage
pytest tests/ -k "edge" --cov=src -v
```

### 2. Create Test Aliases

Add to your `.bashrc` or `.zshrc`:

```bash
alias pytest-all='pytest tests/ -v'
alias pytest-fast='pytest tests/test_react_agent.py::TestTools -v'
alias pytest-cov='pytest tests/ --cov=src --cov-report=html'
alias pytest-debug='pytest tests/ -v -s --tb=short'
```

### 3. Use pytest.ini for Defaults

Create `pytest.ini` in project root:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

### 4. Focus on What Changed

```bash
# If you modified document_parser.py
pytest tests/test_document_parser.py -v

# If you added a new feature
pytest tests/ -k "new_feature" -v

# If fixing a bug
pytest tests/ -k "bug_fix" -v
```

### 5. Performance Profiling

```bash
# Show slowest 10 tests
pytest tests/ --durations=10

# Profile specific file
pytest tests/test_text_classifier.py --durations=0

# Find tests slower than 5 seconds
pytest tests/ --durations=0 | grep -E "[5-9]\.[0-9]+s|[0-9]{2}\.[0-9]+s"
```

---

## 🎯 Test Organization Summary

```
tests/
├── test_document_parser.py      (25 tests) - Extraction & validation
├── test_text_classifier.py      (33 tests) - Classification & routing
├── test_react_agent.py          (25 tests) - Agent & tools
├── test_rag_qa_system.py        (30 tests) - RAG & QA
├── test_multi_step_agent.py     (28 tests) - Multi-step workflows
└── test_application_pipeline.py (35 tests) - Integration & e2e
                                 ───────────
                                 176 tests total
```

---

## ⚡ Quick Reference Card

```bash
# MOST COMMON COMMANDS

# Run everything
pytest tests/ -v

# Run one file
pytest tests/test_document_parser.py -v

# Run one class
pytest tests/test_document_parser.py::TestCustomerOrderValidation -v

# Run one test
pytest tests/test_react_agent.py::TestTools::test_calculator_addition -v

# Run by keyword
pytest tests/ -k "validation" -v

# Stop on failure
pytest tests/ -x

# With coverage
pytest tests/ --cov=src --cov-report=html

# Fast tests only (no API)
pytest tests/test_react_agent.py::TestTools -v
```

---

## 📈 Estimated Runtimes

| Command | Tests | Time | API Cost |
|---------|-------|------|----------|
| Full suite | 176 | ~2-3 min | $0.20-0.30 |
| One file (avg) | ~30 | ~30-45 sec | $0.03-0.05 |
| One class (avg) | ~10 | ~15-20 sec | $0.01-0.02 |
| Tool tests only | 8 | ~1-2 sec | $0.00 |
| Validation tests | 11 | ~5-10 sec | $0.00 |

---

**Happy Testing! 🎉**

For more information on testing patterns, see `src/PATTERNS.md` → Pattern 7: Pytest Testing

