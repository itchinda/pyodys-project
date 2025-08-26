# Contributing to the Numerical ODE Solver

Thank you for your interest in contributing to the Numerical ODE Solver project! We welcome all contributions, from bug reports and feature requests to code changes and documentation improvements.

By participating in this project, you agree to abide by its Code of Conduct.

## How to Contribute

### 1. Reporting Bugs

If you find a bug, please open a new issue on the [GitHub issues page](https://github.com/your-username/your-repo-name/issues). When submitting a bug report, please provide:
* A clear and concise description of the bug.
* The steps to reproduce the behavior.
* The expected behavior.
* Any relevant code snippets or error messages.

### 2. Suggesting Features

We are always open to new ideas for improving the solver. If you have a feature in mind, please open a new issue and describe the feature, its potential use cases, and why you believe it would be a valuable addition.

### 3. Making Code Changes

#### Step 1: Fork the Repository

First, fork the repository to your own GitHub account. This creates a personal copy of the project where you can make changes.

#### Step 2: Clone the Repository

Clone your forked repository to your local machine:
```bash
git clone [https://github.com/itchinda/edo](https://github.com/itchinda/edo)
cd edo
```
#### Step 3: Install in Editable Mode

Install the project and its dependencies in editable mode. This allows you to work on the code directly without having to reinstall it after every change.
```bash
pip install -e .
```

#### Step 4: Create a New Branch
Create a new branch for your feature or bug fix. Use a descriptive name (e.g., `feature/add-new-scheme` or `fix/lorenz-bug`).
```bash
git checkout -b your-branch-name
```

#### Step 5: Write Your Code
Make your code changes. Please ensure your code follows the project's existing style.

#### Step 6: Test Your Changes
Before submitting your changes, run the test suite to ensure you haven't introduced any regressions.
```bash
python -m pytest -v
```

#### Step 7: Commit and Push
Commit your changes with a clear and concise commit message.
```bash
git add .
git commit -m "feat: Add a new Dormand-Prince method"
git push origin your-branch-name
```

#### Step 8: Create a Pull Request
Open a pull request from your new branch to the main branch of the original repository. In the pull request description, please reference any related issues and provide a summary of your changes.

#### 4. Code of Conduct
Please review and adhere to the project's Code of Conduct. We are committed to fostering a welcoming and inclusive community.