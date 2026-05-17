# Contributing to Power Data AI

First off, thank you for considering contributing to Power Data AI! 🎉

Power Data AI is built to empower data-driven decision-making, and your contributions help make that mission stronger.

## Code of Conduct

This project adheres to a code of professionalism and respect. By participating, you are expected to uphold this code.

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates.

**When filing a bug report, include:**
- Clear, descriptive title
- Exact steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- Your environment (OS, Python version, browser)

### 💡 Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues.

**Include:**
- Clear description of the enhancement
- Why this would be useful
- Examples of how it would work

### 🔧 Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Make your changes** following our style guide
3. **Add tests** if you've added functionality
4. **Update documentation** as needed
5. **Ensure tests pass** and code is formatted
6. **Submit a pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/powerdata-ai.git
cd powerdata-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Check code formatting
black powerdata/ app.py
flake8 powerdata/ app.py
```

## Style Guide

### Python Code
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100)
- Use type hints where appropriate
- Write docstrings for all public functions/classes

Example:
```python
def analyze_data(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Analyze a specific column in the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
        column (str): Column name to focus on
    
    Returns:
        Dict containing analysis results
    """
    # Your code here
```

### Commit Messages
Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
feat: add correlation threshold parameter
fix: resolve missing data handling bug
docs: update README with new examples
style: format code with black
refactor: simplify data loading logic
test: add unit tests for analyzer
chore: update dependencies
```

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring

## Testing

All new features should include tests.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=powerdata

# Run specific test file
pytest tests/test_analyzer.py
```

## Documentation

- Update README.md if you change functionality
- Add docstrings to new functions/classes
- Update CHANGELOG.md with your changes

## Financial Contributions

Power Data AI is created by **Issaka Seogo** at **Seogo Global Impact**. If you'd like to support development:

- ⭐ Star the repository
- 🐛 Report bugs and suggest features
- 📢 Share Power Data AI with others
- 💼 Consider Seogo Global Impact for consulting

## Questions?

Feel free to:
- Open a GitHub Discussion
- Email: issaka.seogo@seogoglobalimpacts.com
- Connect on LinkedIn: [Issaka Seogo](https://linkedin.com/in/issaka-seogo)

---

**Thank you for contributing to Power Data AI!**

*Turning your data into direction.*
