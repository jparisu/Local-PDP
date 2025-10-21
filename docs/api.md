# API Reference

Complete API documentation for yourlib.

## Submodule A

### `yourlib.submodule_a.a`

Core functionality for feature attribution analysis.

#### `function_a(x)`

Example function in module A.

**Parameters:**
- `x`: Input parameter

**Returns:**
- Processed result (x * 2)

**Example:**
```python
from yourlib.submodule_a.a import function_a

result = function_a(5)
print(result)  # Output: 10
```

#### `ClassA`

Example class in module A.

##### `__init__(self, value)`

Initialize ClassA.

**Parameters:**
- `value`: Initial value

##### `method_a(self)`

Example method.

**Returns:**
- The stored value

**Example:**
```python
from yourlib.submodule_a.a import ClassA

obj = ClassA(42)
value = obj.method_a()
print(value)  # Output: 42
```

## Submodule B

### `yourlib.submodule_b.b`

Supporting feature attribution functionality.

#### `function_b(y)`

Example function in module B.

**Parameters:**
- `y`: Input parameter

**Returns:**
- Processed result (y + 10)

**Example:**
```python
from yourlib.submodule_b.b import function_b

result = function_b(5)
print(result)  # Output: 15
```

#### `ClassB`

Example class in module B.

##### `__init__(self, data)`

Initialize ClassB.

**Parameters:**
- `data`: Initial data

##### `method_b(self)`

Example method.

**Returns:**
- The stored data

**Example:**
```python
from yourlib.submodule_b.b import ClassB

obj = ClassB([1, 2, 3])
data = obj.method_b()
print(data)  # Output: [1, 2, 3]
```

## Utilities

### `yourlib.utils`

Utility functions for yourlib.

#### `helper_function(a, b)`

Helper function for common operations.

**Parameters:**
- `a`: First parameter
- `b`: Second parameter

**Returns:**
- Combined result (a + b)

**Example:**
```python
from yourlib import utils

result = utils.helper_function(5, 10)
print(result)  # Output: 15
```

#### `validate_input(data)`

Validate input data.

**Parameters:**
- `data`: Data to validate

**Returns:**
- `True` if valid, `False` otherwise

**Example:**
```python
from yourlib import utils

is_valid = utils.validate_input("data")
print(is_valid)  # Output: True

is_valid = utils.validate_input(None)
print(is_valid)  # Output: False
```

#### `process_data(data)`

Process input data.

**Parameters:**
- `data`: Data to process

**Returns:**
- Processed data

**Raises:**
- `ValueError`: If input data is invalid

**Example:**
```python
from yourlib import utils

processed = utils.process_data({"key": "value"})
print(processed)  # Output: {'key': 'value'}

# This will raise ValueError
try:
    utils.process_data(None)
except ValueError as e:
    print(e)  # Output: Invalid input data
```

## Type Hints

This library uses Python type hints where applicable. For full type safety, consider using a type checker like `mypy`:

```bash
mypy yourlib
```

## Error Handling

All functions and methods in this library raise standard Python exceptions where appropriate:

- `ValueError`: For invalid input values
- `TypeError`: For incorrect types
- `AttributeError`: For missing attributes

Always wrap function calls in appropriate exception handlers in production code.
