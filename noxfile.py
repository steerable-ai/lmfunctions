import nox

nox.options.sessions = "lint", "types", "tests"


@nox.session(python=["3.11"])
def lint(session) -> None:
    session.install("pip", "--upgrade", ".")
    session.install("black", ".")
    session.install("black[jupyter]", ".")
    session.install("ruff", ".")
    session.run("black", "--check", ".")
    session.run("ruff", "check", ".")


@nox.session(python=["3.11"])
def types(session) -> None:
    session.install("pip", "--upgrade", ".")
    session.install("mypy", ".")
    session.run(
        "mypy",
        "--ignore-missing-imports",
        "--follow-imports",
        "skip",
        "--install-types",
        "--non-interactive",
        "src",
    )


@nox.session(python=["3.10", "3.11", "3.12"])
def tests(session) -> None:
    session.install("pip", "--upgrade", ".")
    session.install("pytest", ".")
    session.install("pytest-mock", ".")
    session.install("pytest-asyncio", ".")
    session.install("pytest-cov", ".")
    session.install("pytest-xdist", ".")
    session.run(
        "pytest",
        "--verbose",
        "--full-trace",
        "--cov-report",
        "xml:coverage.xml",
        "--cov-report",
        "term",
        "--cov-report",
        "html:htmlcov",
        "--cov=lmfunctions",
        "--cov=tests",
    )
