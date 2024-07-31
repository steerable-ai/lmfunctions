import tempfile

import nox

nox.options.sessions = "lint", "types", "safety", "tests"


@nox.session(python=["3.11"])
def lint(session) -> None:
    session.install("black", ".")
    session.install("black[jupyter]", ".")
    session.install("ruff", ".")
    session.run("black", "--check", ".")
    session.run("ruff", "check", ".")


@nox.session(python=["3.11"])
def types(session) -> None:
    session.run("pip", "install", "--upgrade", "pip")
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


@nox.session(python=["3.11"])
def safety(session):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--with",
            "dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install("safety")
        session.run("safety", "check", f"--file={requirements.name}", "--full-report")


@nox.session(python=["3.10", "3.11", "3.12"])
def tests(session) -> None:
    session.install("llama-cpp-python==0.2.83", ".")
    session.install("litellm", ".")
    session.install("transformers[torch]", ".")
    session.install("fastapi", ".")
    session.install("uvicorn", ".")
    session.install("coverage", ".")
    session.install("pytest", ".")
    session.install("pytest-mock", ".")
    session.install("pytest-asyncio", ".")
    session.install("pytest-cov", ".")
    session.run(
        "pytest",
        "--cov-report",
        "term",
        "--cov-report",
        "html:htmlcov",
        "--cov=lmfunctions",
        "--cov=tests",
    )