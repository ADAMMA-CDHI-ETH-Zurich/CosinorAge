# Contributing to CosinorAge

Thank you for your interest in CosinorAge! Contributions of all kinds are welcome: bug reports, questions, documentation improvements, new data handlers and other code changes.

Everyone taking part in this project is expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Asking questions

For questions about installing or using the package, please open a thread in [GitHub Discussions](https://github.com/ADAMMA-CDHI-ETH-Zurich/CosinorAge/discussions) rather than an issue.

## Reporting bugs and requesting features

Please use the [issue templates](https://github.com/ADAMMA-CDHI-ETH-Zurich/CosinorAge/issues/new/choose):

- **Bug report:** include the CosinorAge, Python and OS versions, a minimal code example and the full error message.
- **Feature request:** describe the use case and, if possible, how you would like the feature to work.

**Never upload participant-level data** (e.g. UK Biobank, NHANES-linked or study data) to issues, discussions or pull requests. If a bug only occurs with your data, try to reproduce it with the simulated sample in `examples/data/sample_enmo.csv` or with a small synthetic file.

## Setting up a development environment

CosinorAge is tested with Python 3.11.

```bash
git clone https://github.com/ADAMMA-CDHI-ETH-Zurich/CosinorAge.git
cd CosinorAge
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

On macOS, LightGBM needs OpenMP; see the installation notes in the [README](README.md).

## Making changes

1. Fork the repository and create a branch from `main`.
2. Make your changes. Follow the [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html) for new or changed functions.
3. Add or update tests in `tests/` and run the test suite:

   ```bash
   pytest
   ```

4. Check for syntax errors and undefined names:

   ```bash
   flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
   ```

5. Open a pull request against `main` describing what you changed and why. Work in progress is welcome; just mark the pull request as a draft.

New data sources can be supported by implementing a new data handler that inherits from the `DataHandler` base class in `cosinorage/datahandlers/`.

## License

By contributing, you agree that your contributions are licensed under the same [license](LICENSE) as the rest of the repository. If you did not write the code yourself, make sure its license is compatible and include the license information, or obtain permission from the original author.
