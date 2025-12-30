#!/usr/bin/env python3
"""
repo-scout: Analyze a repository before cloning or executing code from it.

This script helps you understand what you're getting into before running
untrusted code. It can analyze:
1. Remote GitHub/GitLab repos (via API) before cloning
2. Local directories after cloning

Usage:
    python repo_scout.py https://github.com/user/repo
    python repo_scout.py /path/to/local/repo
    python repo_scout.py .  # Current directory
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ANSI colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def color(text: str, c: str) -> str:
    """Apply color to text."""
    return f"{c}{text}{Colors.END}"

# Suspicious patterns to look for
SUSPICIOUS_PATTERNS = {
    "network_access": [
        r"requests\.(get|post|put|delete)",
        r"urllib\.(request|urlopen)",
        r"http\.client",
        r"socket\.",
        r"curl\s+",
        r"wget\s+",
        r"nc\s+-",  # netcat
    ],
    "code_execution": [
        r"exec\s*\(",
        r"eval\s*\(",
        r"os\.system\s*\(",
        r"subprocess\.(call|run|Popen)",
        r"__import__\s*\(",
        r"\bsh\b.*-c",
        r"bash\s+-c",
    ],
    "file_operations": [
        r"open\s*\([^)]*['\"]w",  # write mode
        r"shutil\.(rmtree|move|copy)",
        r"os\.(remove|unlink|rmdir)",
        r"pathlib.*unlink",
        r"rm\s+-rf",
    ],
    "environment_access": [
        r"os\.environ",
        r"getenv\s*\(",
        r"\.env\b",
        r"API_KEY|SECRET|TOKEN|PASSWORD",
    ],
    "privilege_escalation": [
        r"sudo\s+",
        r"chmod\s+777",
        r"chown\s+root",
        r"setuid",
    ],
    "obfuscation": [
        r"base64\.(b64decode|decode)",
        r"codecs\.decode",
        r"\\x[0-9a-fA-F]{2}",  # hex escapes
        r"chr\s*\(\s*\d+\s*\)",  # chr() calls
    ],
    "crypto_mining": [
        r"stratum\+tcp",
        r"xmrig|minerd|cgminer",
        r"coinhive|cryptoloot",
    ],
}

# File extensions to analyze
CODE_EXTENSIONS = {
    '.py': 'Python',
    '.js': 'JavaScript',
    '.ts': 'TypeScript',
    '.sh': 'Shell',
    '.bash': 'Bash',
    '.rb': 'Ruby',
    '.go': 'Go',
    '.rs': 'Rust',
    '.c': 'C',
    '.cpp': 'C++',
    '.h': 'C Header',
    '.java': 'Java',
    '.php': 'PHP',
    '.pl': 'Perl',
    '.lua': 'Lua',
}

# Files that often contain sensitive/important info
IMPORTANT_FILES = [
    'README.md', 'README.rst', 'README.txt', 'README',
    'LICENSE', 'LICENSE.md', 'LICENSE.txt',
    'SECURITY.md', 'CONTRIBUTING.md',
    'setup.py', 'setup.cfg', 'pyproject.toml',
    'package.json', 'Cargo.toml', 'go.mod',
    'Makefile', 'CMakeLists.txt',
    'Dockerfile', 'docker-compose.yml',
    '.github/workflows/*.yml',
    'requirements.txt', 'requirements-dev.txt',
]


def fetch_github_info(owner: str, repo: str) -> Optional[dict]:
    """Fetch repository info from GitHub API."""
    try:
        import urllib.request
        url = f"https://api.github.com/repos/{owner}/{repo}"
        req = urllib.request.Request(url, headers={'User-Agent': 'repo-scout'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"  {color('Warning:', Colors.YELLOW)} Could not fetch GitHub info: {e}")
        return None


def fetch_github_contents(owner: str, repo: str, path: str = "") -> Optional[list]:
    """Fetch directory contents from GitHub API."""
    try:
        import urllib.request
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        req = urllib.request.Request(url, headers={'User-Agent': 'repo-scout'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception:
        return None


def analyze_remote_repo(url: str) -> dict:
    """Analyze a remote repository before cloning."""
    print(f"\n{color('🔍 Analyzing Remote Repository', Colors.BOLD)}")
    print(f"   URL: {color(url, Colors.CYAN)}\n")

    results = {
        "type": "remote",
        "url": url,
        "warnings": [],
        "info": {},
    }

    # Parse GitHub URL
    parsed = urlparse(url)
    if 'github.com' in parsed.netloc:
        path_parts = parsed.path.strip('/').replace('.git', '').split('/')
        if len(path_parts) >= 2:
            owner, repo = path_parts[0], path_parts[1]

            # Fetch repo info
            info = fetch_github_info(owner, repo)
            if info:
                results["info"] = {
                    "name": info.get("name"),
                    "description": info.get("description"),
                    "owner": info.get("owner", {}).get("login"),
                    "stars": info.get("stargazers_count"),
                    "forks": info.get("forks_count"),
                    "open_issues": info.get("open_issues_count"),
                    "created": info.get("created_at"),
                    "updated": info.get("updated_at"),
                    "language": info.get("language"),
                    "license": info.get("license", {}).get("name") if info.get("license") else None,
                    "archived": info.get("archived"),
                    "fork": info.get("fork"),
                }

                print(f"  {color('Repository Info:', Colors.BOLD)}")
                print(f"    Name: {info.get('full_name')}")
                print(f"    Description: {info.get('description', 'N/A')}")
                print(f"    ⭐ Stars: {info.get('stargazers_count', 0):,}")
                print(f"    🍴 Forks: {info.get('forks_count', 0):,}")
                print(f"    📋 Open Issues: {info.get('open_issues_count', 0)}")
                print(f"    📝 Language: {info.get('language', 'N/A')}")
                print(f"    📄 License: {info.get('license', {}).get('name', 'None')}")
                print(f"    📅 Created: {info.get('created_at', 'N/A')[:10]}")
                print(f"    📅 Last Updated: {info.get('updated_at', 'N/A')[:10]}")

                # Warnings
                if info.get("archived"):
                    results["warnings"].append("Repository is ARCHIVED")
                if info.get("fork"):
                    results["warnings"].append("This is a FORK, not the original repo")
                if info.get("stargazers_count", 0) < 10:
                    results["warnings"].append("Low star count - may be new or unmaintained")
                if not info.get("license"):
                    results["warnings"].append("No license specified")

                # Check for README
                contents = fetch_github_contents(owner, repo)
                if contents:
                    file_names = [f["name"] for f in contents if f["type"] == "file"]
                    if not any(f.lower().startswith("readme") for f in file_names):
                        results["warnings"].append("No README file found")

                    print(f"\n  {color('Root Directory Contents:', Colors.BOLD)}")
                    for item in sorted(contents, key=lambda x: (x["type"] != "dir", x["name"])):
                        icon = "📁" if item["type"] == "dir" else "📄"
                        print(f"    {icon} {item['name']}")
    else:
        print(f"  {color('Note:', Colors.YELLOW)} Non-GitHub URL - limited analysis available")
        results["warnings"].append("Cannot analyze non-GitHub repos remotely. Clone first.")

    return results


def analyze_local_repo(path: str) -> dict:
    """Analyze a local repository/directory."""
    print(f"\n{color('🔍 Analyzing Local Directory', Colors.BOLD)}")
    print(f"   Path: {color(os.path.abspath(path), Colors.CYAN)}\n")

    results = {
        "type": "local",
        "path": os.path.abspath(path),
        "warnings": [],
        "suspicious_findings": defaultdict(list),
        "stats": {},
    }

    path = Path(path)
    if not path.exists():
        print(f"  {color('Error:', Colors.RED)} Path does not exist!")
        return results

    # Check if it's a git repo
    git_dir = path / ".git"
    if git_dir.exists():
        print(f"  {color('✓', Colors.GREEN)} Git repository detected")

        # Get git info
        try:
            remote = subprocess.run(
                ["git", "-C", str(path), "remote", "-v"],
                capture_output=True, text=True
            )
            if remote.stdout:
                print(f"  {color('Remote:', Colors.BOLD)}")
                for line in remote.stdout.strip().split('\n')[:2]:
                    print(f"    {line}")
        except Exception:
            pass

    # Collect file stats
    file_counts = Counter()
    total_lines = 0
    all_files = []

    # Directories to skip
    skip_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv',
                 '.tox', '.pytest_cache', 'dist', 'build', '.eggs'}

    for root, dirs, files in os.walk(path):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            filepath = Path(root) / file
            relative = filepath.relative_to(path)
            all_files.append(relative)

            ext = filepath.suffix.lower()
            if ext in CODE_EXTENSIONS:
                file_counts[CODE_EXTENSIONS[ext]] += 1

                # Analyze file contents for suspicious patterns
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')
                        total_lines += len(lines)

                        # Check for suspicious patterns
                        for category, patterns in SUSPICIOUS_PATTERNS.items():
                            for pattern in patterns:
                                matches = re.finditer(pattern, content, re.IGNORECASE)
                                for match in matches:
                                    line_num = content[:match.start()].count('\n') + 1
                                    results["suspicious_findings"][category].append({
                                        "file": str(relative),
                                        "line": line_num,
                                        "match": match.group()[:50],
                                    })
                except Exception:
                    pass

    results["stats"] = {
        "total_files": len(all_files),
        "code_files": dict(file_counts),
        "total_lines": total_lines,
    }

    # Print stats
    print(f"\n  {color('File Statistics:', Colors.BOLD)}")
    print(f"    Total files: {len(all_files)}")
    print(f"    Total lines of code: {total_lines:,}")
    print(f"\n  {color('Languages:', Colors.BOLD)}")
    for lang, count in file_counts.most_common(10):
        bar = "█" * min(count, 30)
        print(f"    {lang:15} {count:5} {color(bar, Colors.BLUE)}")

    # Check for important files
    print(f"\n  {color('Important Files:', Colors.BOLD)}")
    for pattern in IMPORTANT_FILES:
        if '*' in pattern:
            matches = list(path.glob(pattern))
            for m in matches:
                print(f"    ✓ {m.relative_to(path)}")
        else:
            if (path / pattern).exists():
                print(f"    ✓ {pattern}")

    # Print suspicious findings
    if results["suspicious_findings"]:
        print(f"\n  {color('⚠️  Suspicious Patterns Found:', Colors.YELLOW + Colors.BOLD)}")
        for category, findings in results["suspicious_findings"].items():
            if findings:
                print(f"\n    {color(category.replace('_', ' ').title(), Colors.YELLOW)} ({len(findings)} occurrences):")
                # Show first 5 examples
                for finding in findings[:5]:
                    print(f"      • {finding['file']}:{finding['line']} - {finding['match']}")
                if len(findings) > 5:
                    print(f"      ... and {len(findings) - 5} more")
    else:
        print(f"\n  {color('✓ No suspicious patterns detected', Colors.GREEN)}")

    # Check for potentially dangerous files
    dangerous_extensions = {'.exe', '.dll', '.so', '.dylib', '.bin', '.msi'}
    dangerous_files = [f for f in all_files if f.suffix.lower() in dangerous_extensions]
    if dangerous_files:
        results["warnings"].append(f"Found {len(dangerous_files)} binary/executable files")
        print(f"\n  {color('⚠️  Binary/Executable Files:', Colors.YELLOW)}")
        for f in dangerous_files[:10]:
            print(f"    • {f}")

    return results


def print_summary(results: dict):
    """Print a summary with recommendations."""
    print(f"\n{color('=' * 60, Colors.BOLD)}")
    print(f"{color('SUMMARY', Colors.BOLD)}")
    print(f"{color('=' * 60, Colors.BOLD)}")

    if results["warnings"]:
        print(f"\n{color('⚠️  Warnings:', Colors.YELLOW + Colors.BOLD)}")
        for warning in results["warnings"]:
            print(f"  • {warning}")

    if results.get("suspicious_findings"):
        total = sum(len(v) for v in results["suspicious_findings"].values())
        if total > 0:
            print(f"\n{color('🔍 Suspicious Patterns:', Colors.YELLOW)}")
            print(f"  Found {total} potentially suspicious code patterns.")
            print(f"  Review the findings above before running any code.")

    print(f"\n{color('📋 Recommendations:', Colors.CYAN)}")
    if results["type"] == "remote":
        print("  1. Review the repository's README and documentation")
        print("  2. Check recent commits and issues for activity")
        print("  3. Verify the repository owner/organization is legitimate")
        print("  4. Clone to a sandboxed environment first")
    else:
        print("  1. Review any suspicious patterns found above")
        print("  2. Check setup.py/package.json for install hooks")
        print("  3. Review shell scripts before executing")
        print("  4. Use a virtual environment or container")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a repository before cloning or executing code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://github.com/user/repo    # Analyze remote repo
  %(prog)s /path/to/local/repo             # Analyze local directory
  %(prog)s .                               # Analyze current directory
        """
    )
    parser.add_argument("target", help="Repository URL or local path")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Minimal output")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")

    args = parser.parse_args()
    target = args.target

    # Determine if remote or local
    if target.startswith(('http://', 'https://', 'git@')):
        results = analyze_remote_repo(target)
    else:
        results = analyze_local_repo(target)

    if args.json:
        # Convert defaultdict to regular dict for JSON
        if "suspicious_findings" in results:
            results["suspicious_findings"] = dict(results["suspicious_findings"])
        print(json.dumps(results, indent=2, default=str))
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
