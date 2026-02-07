# Test Suite Documentation

## Overview

This document provides a comprehensive overview of the test suite added to the brimfile project.

## Test Suite Statistics

- **Total Tests**: 98
- **Test Files**: 7 (plus 1 original demo file)
- **Code Coverage**: 65% overall
  - `__init__.py`: 100%
  - `constants.py`: 100%
  - `units.py`: 100%
  - `metadata.py`: 97%
  - `file.py`: 83%
  - `utils.py`: 73%
  - `data.py`: 64%
  - `file_abstraction.py`: 43%

## Test Organization

### Unit Tests

1. **test_file.py** (24 tests)
   - File creation and opening
   - Read-only mode handling
   - Data group operations
   - Store type handling
   - File lifecycle management

2. **test_data.py** (11 tests)
   - Data properties (name, index, parameters)
   - Spectrum retrieval at different coordinates
   - Metadata access
   - Analysis results management
   - Data creation with different dimensions

3. **test_metadata.py** (20 tests)
   - Metadata item creation
   - Adding metadata (experiment, optics, etc.)
   - Retrieving metadata
   - Dictionary conversions
   - Metadata types
   - Local vs global metadata

4. **test_analysis_results.py** (17 tests)
   - Analysis results properties
   - Peak types and quantities
   - Image retrieval (shift, width)
   - Units handling
   - Pixel quantity retrieval
   - Fit models

5. **test_utils.py** (13 tests)
   - Path concatenation
   - Variable to singleton conversion
   - NumPy array type optimization
   - Array operations

### Integration Tests

6. **test_integration.py** (13 tests)
   - Complete read/write workflows
   - Multiple data groups and analysis results
   - Data consistency across operations
   - Read-only behavior
   - Edge cases (empty files, single points, large arrays)
   - File lifecycle management

## Test Fixtures

Defined in `conftest.py`:

- **sample_data**: Generated Lorentzian spectral data
- **simple_brim_file**: Pre-populated brim file for testing
- **empty_brim_file**: Empty brim file for creation tests

Note: Tests use pytest's built-in `tmp_path` fixture for temporary directories.

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/brimfile --cov-report=html

# Run specific test file
pytest tests/test_file.py -v

# Run specific test class
pytest tests/test_file.py::TestFileCreation -v
```

### CI/CD Integration

A GitHub Actions workflow (`test.yml`) has been added to run tests automatically on:
- Push to main/develop branches
- Pull requests to main/develop branches
- Multiple OS: Ubuntu, Windows, macOS
- Multiple Python versions: 3.11, 3.12

## Test Coverage Details

### Well-Covered Areas (>80%)
- File operations and management
- Metadata handling
- Constants and utilities
- Basic data operations

### Areas for Improvement (<70%)
- `file_abstraction.py` (43%): Low-level file abstraction layer
- `data.py` (64%): Complex data manipulation operations

These lower-coverage areas involve:
- Advanced file operations
- Edge cases in data conversion
- Asynchronous operations
- Error handling in unusual scenarios

## Key Test Scenarios Covered

1. **File Operations**
   - Creating new files with different store types
   - Opening existing files in different modes
   - Handling non-existent files
   - File validation

2. **Data Management**
   - Creating data groups with spectral data
   - Retrieving spectra at specific coordinates
   - Managing multiple data groups
   - Handling different data dimensions

3. **Metadata Operations**
   - Adding various metadata types
   - Retrieving metadata items
   - Local vs global metadata
   - Updating existing metadata

4. **Analysis Results**
   - Creating analysis results with different fit models
   - Retrieving images for different quantities
   - Getting values at specific pixels
   - Listing available peak types and quantities

5. **Integration Workflows**
   - Complete create → write → read cycles
   - Multiple sequential operations
   - Data consistency checks
   - Read-only vs write mode behavior

## Known Test Warnings

Some expected warnings during test execution:
- "No units provided for X; None is assumed" - Normal when metadata lacks units
- "Cannot close the file" - Rare edge case, handled gracefully

## Future Enhancements

Potential areas for additional testing:
1. Advanced data conversion scenarios
2. Network/remote storage operations
3. Performance/stress testing
4. Export functionality (OME-TIFF)
5. Converter module testing
6. More extensive error handling scenarios

## Maintenance

When adding new features:
1. Add corresponding tests in the appropriate test file
2. Ensure tests follow existing patterns
3. Use fixtures for common setup
4. Add docstrings to explain test purpose
5. Run full test suite before committing
6. Update this documentation if needed

## Dependencies

Test dependencies (installed separately):
- pytest >= 9.0.2
- pytest-cov >= 7.0.0

Runtime dependencies (from pyproject.toml):
- numpy
- zarr >= 3.1.1
