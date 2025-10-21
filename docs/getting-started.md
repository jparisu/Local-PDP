# Getting Started

This guide will help you get started with yourlib.

## Installation

### From PyPI

```bash
pip install yourlib
```

### From Source

```bash
git clone https://github.com/jparisu/Local-PDP.git
cd Local-PDP
pip install -e .
```

## Basic Usage

### Using Submodule A

```python
from yourlib.submodule_a.a import function_a, ClassA

# Use function_a
result = function_a(10)
print(f"Result: {result}")  # Output: Result: 20

# Create a ClassA instance
obj = ClassA(value=100)
value = obj.method_a()
print(f"Value: {value}")  # Output: Value: 100
```

### Using Submodule B

```python
from yourlib.submodule_b.b import function_b, ClassB

# Use function_b
result = function_b(5)
print(f"Result: {result}")  # Output: Result: 15

# Create a ClassB instance
obj = ClassB(data=[1, 2, 3])
data = obj.method_b()
print(f"Data: {data}")  # Output: Data: [1, 2, 3]
```

### Using Utilities

```python
from yourlib import utils

# Use helper function
result = utils.helper_function(5, 10)
print(f"Result: {result}")  # Output: Result: 15

# Validate input
is_valid = utils.validate_input("some data")
print(f"Is valid: {is_valid}")  # Output: Is valid: True

# Process data
processed = utils.process_data({"key": "value"})
print(f"Processed: {processed}")  # Output: Processed: {'key': 'value'}
```

## Next Steps

- Read the [API Reference](api.md) for detailed documentation
- Check out the [examples](https://github.com/jparisu/Local-PDP/tree/main/examples) (coming soon)
- Contribute to the project by following the [Contributing Guide](../CONTRIBUTING.md)

## Requirements

- Python 3.8 or higher
- Dependencies as listed in `pyproject.toml`

## Troubleshooting

### Import Errors

If you encounter import errors, make sure yourlib is properly installed:

```bash
pip install -e .
```

### Version Issues

Check your Python version:

```bash
python --version
```

Make sure you're using Python 3.8 or higher.

## Getting Help

- Open an issue on [GitHub](https://github.com/jparisu/Local-PDP/issues)
- Check existing documentation
- Review the examples directory
