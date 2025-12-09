# Feature Attribution Explainability Python Library

[![Docs](https://readthedocs.org/projects/faex/badge/?version=latest)](https://faex.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/jparisu/faex/actions/workflows/ci.yml/badge.svg)](https://github.com/jparisu/faex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jparisu/faex/branch/main/graph/badge.svg)](https://codecov.io/gh/jparisu/faex)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jparisu/faex/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)


Feature Attribution Model-Agnostic **Python library** for Machine Learning **Explainability**.

This repository implements `faex` python library with several feature attribution agnostic-model methods for explaining predictions of machine learning models.
Mainly, it focuses on **Local-PDP** (Local Partial Dependence Plots) method, which is designed to provide reliable explanations in any dataset, removing PDP independence assumptions.

📘 **Documentation:** [https://faex.readthedocs.io](https://faex.readthedocs.io)


## Installation

You can install `faex` via `pip` from github repository:

```bash
pip install git+https://github.com/jparisu/faex.git
```
