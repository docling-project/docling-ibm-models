## Contributing In General
Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

For more details on the contributing guidelines head to the Docling Project [community repository](https://github.com/docling-project/community).

## Developing

### Usage of uv

We use [uv](https://docs.astral.sh/uv/) as package and project manager.

#### Installation

To install `uv`, check the documentation on [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

#### Create an environment and sync it

You can use the `uv sync` to create a project virtual environment (if it does not already exist) and sync
the project's dependencies with the environment.

```bash
uv sync
```

#### Use a specific Python version (optional)

If you need to work with a specific version of Python, you can create a new virtual environment for that version
and run the sync command:

```bash
uv venv --python 3.12
uv sync
```

More detailed options are described on the [Using Python environments](https://docs.astral.sh/uv/pip/environments/) documentation.

#### Add a new dependency

Simply use the `uv add` command. The `pyproject.toml` and `uv.lock` files will be updated.

```bash
uv add [OPTIONS] <PACKAGES|--requirements <REQUIREMENTS>>
```

### Code sytle guidelines

We use the following tools to enforce code style:

- [Ruff](https://docs.astral.sh/ruff/), to format code and sort imports
- [ty](https://docs.astral.sh/ty/), as static type checker

A set of styling checks is defined and managed through [prek](https://github.com/j178/prek), a fast drop-in replacement for the [pre-commit](https://pre-commit.com/) framework. To ensure that those scripts run automatically before a commit is finalized, install the hooks on your local repository:

```bash
make hooks-install
```

To run the checks on-demand, type:

```bash
make validate      # hooks on the changed files
make validate-all  # hooks on all files
make check         # read-only checks
make fix           # apply the Ruff formatter and fixers
```

Note: Checks like the Ruff formatter will _fail_ if they modify files. In these cases, `git add` the modified files and `git commit` again.

Pull requests also run a fast lint and typing check on the changed files. This check does not need a maintainer approval to run.

