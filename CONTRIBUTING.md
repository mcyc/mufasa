# Contributing to MUFASA

Here are the guidelines for contributing to MUFASA. The best place to start contributing is by 
initiating or joining a discussion via a GitHub issue.

## Implementing new Spectral Models:

New spectral models should be implemented in the `spec_models.SpecModels` module, with their associated
molecular constants implemented inside the `spec_models.m_constants` module. Currently, there are
the `spec_models.HyperfineModel.HyperfineModel` and `spec_models.BaseModel.BaseModel` classes
to create spectral models with and without hyperfine structures, respectively, via inheritance.

Below is a hypothetical
example of how such an implementation would look inside `SpecModels.py`, without the docstrings:
   ```
   class HNC_Model(HyperfineModel):
      from .m_constants import hnc as _hnc_constants
      _molecular_constants = _hnc_constants
      
      def __init__(self, line_names=['onezero']):
         super().__init__(line_names=line_names)
   ```

where its accompanying code in `m_constants.py` would look like:
   ```
   hnc_constants = dict(
      line_names = ['onezero'],              # more than a single transition is permitted
      freq_dict = {'onezero':  90.6636e9},   # rest frequency in Hz
      voff_lines_dict = {
         'onezero': [-voff_0, ..., voff_n],  # velocity offsets of hyperfine lines
         },
      tau_wts_dict = {
         'onezero': [tau_hf0, ..., tau_hfn]  # relative tau weight of the hyperfine lines
         }
   )
   ```

## General Code Contributions

1. **Planning**:
    - Initiate or join a GitHub issue to discuss the plans to contribute

2. **Coding**:
    - Use numpy-style docstrings for documentation 
    - Run tests locally before pushing commits.

3. **Commit Changes**:
    - Use clear and descriptive commit messages.
    - Example:
        ```
        Fix Sphinx import errors and update Makefile
        ```

4. **Submit a Pull Request**:
    - Push your changes to a new branch and open a pull request on GitHub.

---


## Building Documentation

Before making a pull request on GitHub, please ensure the documentation can be built properly by Read
the Docs (RTD). To test build the documentation locally:

1. Navigate to the `docs` directory:
    ```bash
    cd docs
    ```

2. Build the HTML documentation:
    ```bash
    make all
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


