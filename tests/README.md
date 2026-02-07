# Brimfile Tests

This directory contains the comprehensive test suite for the brimfile package.

## Test Structure

The test suite is organized into multiple test files, each focusing on different components:

- **`conftest.py`**: Shared pytest fixtures and test utilities
- **`test_file.py`**: Tests for the File class (file creation, opening, data group management)
- **`test_data.py`**: Tests for the Data class (spectrum retrieval, metadata access, analysis results)
- **`test_metadata.py`**: Tests for the Metadata class (adding/retrieving metadata, types, conversions)
- **`test_analysis_results.py`**: Tests for the AnalysisResults class (image retrieval, quantities, peak types)
- **`test_integration.py`**: Integration tests for complete workflows and edge cases
- **`test_utils.py`**: Tests for utility functions
- **`general.py`**: Original demonstration script (kept for reference)

## Running the Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run all tests with coverage report
pytest tests/ --cov=src/brimfile --cov-report=html

# Run specific test file
pytest tests/test_file.py -v

# Run specific test class
pytest tests/test_file.py::TestFileCreation -v

# Run specific test
pytest tests/test_file.py::TestFileCreation::test_create_file_auto_store -v
```

### Test Configuration

The test configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--verbose",
    "--strict-markers",
    "--tb=short",
    "--import-mode=importlib",
]
```

The `--import-mode=importlib` flag is used as recommended by pytest documentation for more robust test imports.

## Test Coverage

The test suite covers:

- ✅ File operations (creation, opening, closing, validation)
- ✅ Data group management (create, list, retrieve)
- ✅ Spectrum retrieval at different coordinates
- ✅ Metadata operations (add, retrieve, update, local vs global)
- ✅ Analysis results (images, quantities, peak types, units)
- ✅ Integration workflows (complete read/write cycles)
- ✅ Edge cases (empty files, single points, large arrays)
- ✅ Read-only vs write mode behavior
- ✅ Multiple data groups and analysis results
- ✅ Utility functions

## Test Fixtures

The `conftest.py` file provides shared fixtures:

- **`sample_data`**: Generated sample spectral data for testing
- **`simple_brim_file`**: Pre-created brim file with sample data
- **`empty_brim_file`**: Empty brim file for testing creation operations

Note: Tests use pytest's built-in `tmp_path` fixture for temporary directories with automatic cleanup.

## Writing New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py` for files, `Test*` for classes, `test_*` for methods
2. Use descriptive test names that explain what is being tested
3. Organize tests into logical classes
4. Use fixtures from `conftest.py` when possible
5. Clean up any created files (fixtures handle this automatically)
6. Add docstrings to explain complex test scenarios

Example:

```python
class TestNewFeature:
    """Tests for the new feature."""
    
    def test_basic_functionality(self, simple_brim_file):
        """Test basic functionality of the feature."""
        f = brim.File(simple_brim_file)
        # Test code here
        f.close()
```

## Known Test Warnings

Some tests may produce warnings that are expected:

- "No units provided for X; None is assumed" - Expected when metadata items lack units
- "Cannot close the file" - May occur in some edge cases, handled gracefully

These warnings do not indicate test failures and are part of normal operation.

## Continuous Integration

The tests are designed to run in CI/CD environments. To integrate with GitHub Actions:

```yaml
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ --cov=src/brimfile --cov-report=xml
```

## Test Metrics

Current test statistics:
- **Total tests**: 98
- **Test files**: 7
- **Code coverage**: Run `pytest --cov=src/brimfile --cov-report=html` to generate coverage report
