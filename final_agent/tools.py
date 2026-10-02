import subprocess
import tempfile
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults


web_search = TavilySearchResults(
    name="web_search",
    description="Search the web for relevant sources and snippets.",
    max_results=5,
    search_depth="advanced",
    include_answer=True,
)


@tool
def fetch_webpage(url: str) -> str:
    """Fetch a webpage and return readable text."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        return f"ERROR: failed to fetch webpage: {exc}"

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = soup.get_text("\n", strip=True)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    return f"Title: {title}\nURL: {url}\n\n{text[:12000]}"


@tool
def read_text_file(path: str) -> str:
    """Read a local text file."""
    file_path = Path(path)

    if not file_path.exists():
        return f"ERROR: file does not exist: {path}"

    try:
        return file_path.read_text(encoding="utf-8")[:12000]
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1")[:12000]
    except Exception as exc:
        return f"ERROR: failed to read file: {exc}"


@tool
def python_exec(code: str) -> str:
    """Execute Python code and return stdout/stderr."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        encoding="utf-8",
        delete=False,
    ) as file:
        file.write(code)
        temp_path = file.name

    try:
        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return f"ERROR: python execution failed: {exc}"
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass

    return (
        f"STDOUT:\n{result.stdout.strip()}\n\n"
        f"STDERR:\n{result.stderr.strip()}\n\n"
        f"EXIT CODE: {result.returncode}"
    )