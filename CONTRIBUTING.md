# Contributing to MUFASA

Here are the guidelines on how to contribute to MUFASA.

## Setting Up Your Environment

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/<your-username>/mufasa.git
    cd mufasa
    ```

2. **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies**:
    - For development:
        ```bash
        pip install -e .[dev]
        ```
    - For documentation:
        ```bash
        pip install -r docs/requirements.txt
        ```

## Building Documentation

To build the documentation locally:

1. Navigate to the `docs` directory:
    ```bash
    cd docs
    ```

2. Build the HTML documentation:
    ```bash
    make html
    ```

3. View the output in `docs/build/html/index.html`.

### Common Issues

- **Cache Errors**: If you encounter pickle or stale file issues, clear Sphinx’s cache:
    ```bash
    rm -rf docs/source/.doctrees docs/build
    ```

- **Module Import Errors**: Ensure the Mufasa package is installed in editable mode:
    ```bash
    pip install -e .
    ```

## Code Contributions

1. **Follow PEP 8**:
    - Ensure your code adheres to [PEP 8](https://pep8.org) standards.
    - Run linting checks:
        ```bash
        flake8 mufasa
        ```

2. **Write Tests**:
    - Add or update tests in the `tests/` directory.
    - Run tests locally:
        ```bash
        pytest
        ```

3. **Commit Changes**:
    - Use clear and descriptive commit messages.
    - Example:
        ```
        Fix Sphinx import errors and update Makefile
        ```

4. **Submit a Pull Request**:
    - Push your changes to a new branch and open a pull request on GitHub.

---

