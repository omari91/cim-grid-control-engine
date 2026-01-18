# Contributing to CIM Grid Control Engine

Thank you for your interest in contributing to the CIM Grid Control Engine! This project aims to advance distribution grid analysis through physics-based modeling and intelligent control systems. We welcome contributions from the power systems community, software developers, and researchers.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Guidelines](#development-guidelines)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Community and Support](#community-and-support)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you are expected to uphold this standard. Please be respectful, collaborative, and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Understanding of power systems fundamentals
- Familiarity with PandaPower library (optional but helpful)
- Knowledge of IEC 61970 CIM standards (for data model contributions)

### Setting Up Your Development Environment

1. **Fork the repository**
   ```bash
   # Fork via GitHub UI, then clone your fork
   git clone https://github.com/YOUR_USERNAME/cim-grid-control-engine.git
   cd cim-grid-control-engine
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the test suite**
   ```bash
   python main.py
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include:

- **Clear descriptive title**
- **Detailed description** of the issue
- **Steps to reproduce** the behavior
- **Expected vs. actual behavior**
- **System information** (OS, Python version, dependency versions)
- **Error messages** or stack traces
- **Sample grid data** if applicable (anonymized if necessary)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use case description** - What problem does this solve?
- **Proposed solution** - How should it work?
- **Alternatives considered** - What other approaches did you consider?
- **Impact on existing features** - Will this break anything?
- **Relevant research or standards** - Links to papers, standards, or documentation

### Areas Where Contributions Are Especially Welcome

1. **Load Flow Algorithms**
   - Additional solver implementations (Newton-Raphson, Gauss-Seidel)
   - Performance optimizations for large networks
   - Convergence improvements

2. **Control Strategies**
   - Alternative Volt-VAR control algorithms
   - Coordination strategies for distributed energy resources (DER)
   - Advanced hosting capacity methodologies

3. **CIM Compliance**
   - Enhanced IEC 61970 support
   - Data model validation
   - Import/export utilities for standard formats

4. **Documentation**
   - Code documentation and docstrings
   - Usage examples and tutorials
   - Validation case studies
   - API reference improvements

5. **Testing**
   - Additional test cases
   - Real-world grid validation scenarios
   - Performance benchmarks

## Development Guidelines

### Branch Naming Convention

Use descriptive branch names that reflect the purpose:

- `feature/description` - New features (e.g., `feature/newton-raphson-solver`)
- `fix/description` - Bug fixes (e.g., `fix/voltage-convergence-issue`)
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions

### Commit Messages

Write clear, concise commit messages:

```
Add forward-backward sweep solver optimization

- Implement sparse matrix operations for improved performance
- Add convergence tolerance parameter
- Update documentation with performance benchmarks

Addresses #42
```

Format:
- Use imperative mood ("Add feature" not "Added feature")
- First line: Brief summary (50 characters or less)
- Blank line
- Detailed description (wrap at 72 characters)
- Reference related issues

## Coding Standards

### Python Style Guide

We follow PEP 8 with some project-specific conventions:

- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Naming conventions**:
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`

### Documentation Standards

All public functions and classes must include docstrings:

```python
def calculate_voltage_profile(network, load_data):
    """
    Calculate voltage profile for all buses in the distribution network.
    
    Args:
        network (dict): Network topology and parameters following CIM structure
        load_data (pd.DataFrame): Load profile data with columns ['bus_id', 'p_mw', 'q_mvar']
        
    Returns:
        pd.DataFrame: Voltage magnitude and angle for each bus
        
    Raises:
        ConvergenceError: If power flow does not converge within max iterations
        ValueError: If network topology is invalid
        
    Example:
        >>> network = load_cim_network('grid_model.xml')
        >>> loads = pd.read_csv('load_profile.csv')
        >>> voltages = calculate_voltage_profile(network, loads)
    """
    pass
```

### Code Organization

- Keep functions focused and single-purpose
- Avoid deep nesting (max 3 levels)
- Use meaningful variable names
- Add comments for complex algorithms or non-obvious logic
- Group related functions into modules

## Testing Requirements

### Running Tests

All contributions must include appropriate tests:

```bash
# Run all tests
python main.py

# Run specific validation scenarios
python -c "from main import run_validation; run_validation()"
```

### Test Coverage Expectations

- **New features**: Minimum 80% code coverage
- **Bug fixes**: Include test case that reproduces the bug
- **Algorithm changes**: Include validation against known results

### Test Case Structure

```python
def test_forward_backward_sweep_ieee13():
    """
    Test FBS solver accuracy against IEEE 13-bus test feeder.
    
    Validates:
    - Voltage profile matches reference within tolerance
    - Power flow convergence
    - Loss calculations
    """
    # Arrange
    network = load_ieee_test_feeder(13)
    expected_voltages = load_reference_results('ieee13_voltages.csv')
    
    # Act
    results = forward_backward_sweep(network)
    
    # Assert
    assert_voltage_within_tolerance(results, expected_voltages, tolerance=1e-4)
    assert results['converged'] == True
    assert results['iterations'] < 10
```

## Pull Request Process

### Before Submitting

1. **Update your branch**
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Run all tests** and ensure they pass

3. **Update documentation** to reflect any changes

4. **Review your changes** - check for debug code, commented sections, or unnecessary files

### Submitting the Pull Request

1. Push your branch to your fork
2. Navigate to the original repository
3. Click "New Pull Request"
4. Select your branch
5. Fill out the PR template with:
   - **Description**: What does this PR do?
   - **Motivation**: Why is this change needed?
   - **Testing**: How was this tested?
   - **Breaking changes**: Any backwards incompatibility?
   - **Related issues**: Link to relevant issues

### PR Review Process

- Maintainers will review your code
- Address feedback by pushing new commits
- Once approved, your PR will be merged
- You may be asked to squash commits before merging

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added and passing
- [ ] No merge conflicts
- [ ] Commit messages are clear

## Community and Support

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions, ideas, and general discussion
- **Pull Requests**: Code contributions and reviews

### Getting Help

If you need help:

1. Check existing documentation and issues
2. Search for similar questions in discussions
3. Create a new discussion with:
   - Clear description of what you're trying to do
   - What you've already tried
   - Relevant code snippets or error messages

### Recognition

Contributors are recognized in:
- Release notes for significant contributions
- README acknowledgments
- Git commit history

---

## Additional Resources

### Power Systems Standards

- [IEC 61970 (CIM)](https://en.wikipedia.org/wiki/Common_Information_Model_(electricity))
- [IEEE Test Feeders](https://site.ieee.org/pes-testfeeders/)
- [VDE-AR-N 4105](https://www.vde.com/en/fnn/topics/technical-connection-rules) - Voltage regulation standards

### Recommended Reading

- PandaPower Documentation: https://pandapower.readthedocs.io/
- Distribution System Analysis: Kersting, W.H. "Distribution System Modeling and Analysis"
- Grid Integration: EPRI Distribution System Analysis Guidelines

### Development Tools

- **Linting**: `flake8` or `pylint`
- **Formatting**: `black` (optional)
- **Type checking**: `mypy` (optional)

---

**Thank you for contributing to advancing open-source power systems analysis tools!**

For questions about contributing, open a discussion or reach out through GitHub issues.
