# Getting Started

This guide will help you get started with `faxai`.
First, make sure to install the library: [Installation](installation.md)

## Basic Usage

### Using Submodule A

```python
from faxai.submodule_a.a import function_a, ClassA

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
from faxai.submodule_b.b import function_b, ClassB

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
from faxai import utils

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


### Logging
`faxai` emits logs via the standard `logging` library.

Enable globally:
```python
import logging
logging.basicConfig(level=logging.INFO)
```


## Troubleshooting

### Import Errors

If you encounter import errors, make sure faxai is properly installed:

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

- Open an issue on [GitHub](https://github.com/jparisu/faxai/issues)
- Check existing documentation
- Review the examples directory
