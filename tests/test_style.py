import ast
import io
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".py", ".toml", ".md", ".yml", ".yaml", ".html", ".txt", ".cfg", ".json"}


def tracked() -> list[Path]:
    try:
        output = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        output = ""
    files = [ROOT / line for line in output.splitlines() if line]
    if files:
        return files
    folders = [ROOT / "src", ROOT / "tests", ROOT / ".github"]
    found = [
        p for f in folders for p in f.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    ]
    return found + [p for p in ROOT.iterdir() if p.is_file()]


def test_no_comments_or_docstrings():
    problems = []
    for path in tracked():
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                problems.append(f"{path}:{token.start[0]} comment")
        for node in ast.walk(ast.parse(source)):
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ) and ast.get_docstring(node):
                problems.append(f"{path} docstring in {getattr(node, 'name', 'module')}")
    assert problems == []


def test_tracked_text_is_plain_ascii():
    problems = []
    for path in tracked():
        if path.suffix in TEXT or path.name in {".gitignore", "LICENSE"}:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if any(ord(ch) > 127 for ch in line):
                    problems.append(f"{path}:{number}")
    assert problems == []
