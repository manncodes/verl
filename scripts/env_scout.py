#!/usr/bin/env python3
"""
env-scout: Analyze the current execution environment.

This script helps understand the environment where commands are being executed:
- System information (OS, kernel, architecture)
- User context and permissions
- Container/sandbox detection
- Network capabilities
- File system access
- Available tools and runtimes
- Resource limits

Usage:
    python env_scout.py
    python env_scout.py --json
"""

import argparse
import json
import os
import platform
import pwd
import grp
import resource
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ANSI colors
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

def color(text: str, c: str) -> str:
    return f"{c}{text}{Colors.END}"

def run_cmd(cmd: List[str], timeout: int = 5) -> Tuple[bool, str]:
    """Run a command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return False, ""

def check_command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    success, _ = run_cmd(["which", cmd])
    return success

def get_system_info() -> Dict[str, Any]:
    """Gather system information."""
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
    }

    # Get kernel info on Linux
    if platform.system() == "Linux":
        success, uname = run_cmd(["uname", "-a"])
        if success:
            info["uname"] = uname

    return info

def get_user_info() -> Dict[str, Any]:
    """Get current user information."""
    info = {
        "uid": os.getuid(),
        "gid": os.getgid(),
        "euid": os.geteuid(),
        "egid": os.getegid(),
        "username": pwd.getpwuid(os.getuid()).pw_name,
        "home": str(Path.home()),
        "cwd": os.getcwd(),
        "groups": [],
    }

    # Get group names
    try:
        groups = os.getgroups()
        info["groups"] = [grp.getgrgid(g).gr_name for g in groups]
    except (KeyError, OSError):
        info["groups"] = [str(g) for g in os.getgroups()]

    # Check if root
    info["is_root"] = os.getuid() == 0

    # Check sudo capability
    success, _ = run_cmd(["sudo", "-n", "true"])
    info["has_passwordless_sudo"] = success

    return info

def detect_container() -> Dict[str, Any]:
    """Detect if running inside a container."""
    info = {
        "is_container": False,
        "container_type": None,
        "indicators": [],
    }

    # Check for Docker
    if Path("/.dockerenv").exists():
        info["is_container"] = True
        info["container_type"] = "docker"
        info["indicators"].append("/.dockerenv exists")

    # Check cgroup for docker/container indicators
    try:
        with open("/proc/1/cgroup", "r") as f:
            cgroup = f.read()
            if "docker" in cgroup:
                info["is_container"] = True
                info["container_type"] = "docker"
                info["indicators"].append("docker in /proc/1/cgroup")
            elif "kubepods" in cgroup:
                info["is_container"] = True
                info["container_type"] = "kubernetes"
                info["indicators"].append("kubepods in /proc/1/cgroup")
            elif "lxc" in cgroup:
                info["is_container"] = True
                info["container_type"] = "lxc"
                info["indicators"].append("lxc in /proc/1/cgroup")
    except (FileNotFoundError, PermissionError):
        pass

    # Check for container-specific env vars
    container_env_vars = ["KUBERNETES_SERVICE_HOST", "DOCKER_CONTAINER", "container"]
    for var in container_env_vars:
        if os.environ.get(var):
            info["is_container"] = True
            info["indicators"].append(f"env var {var} is set")

    # Check init process
    try:
        with open("/proc/1/comm", "r") as f:
            init = f.read().strip()
            if init not in ["init", "systemd"]:
                info["indicators"].append(f"PID 1 is {init}")
    except (FileNotFoundError, PermissionError):
        pass

    return info

def detect_sandbox() -> Dict[str, Any]:
    """Detect sandbox/restricted environment indicators."""
    info = {
        "indicators": [],
        "restrictions": [],
    }

    # Check for common sandbox indicators
    sandbox_paths = [
        "/run/sandbox",
        "/var/run/sandbox",
    ]
    for path in sandbox_paths:
        if Path(path).exists():
            info["indicators"].append(f"Sandbox path exists: {path}")

    # Check seccomp
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("Seccomp:"):
                    seccomp_mode = line.split(":")[1].strip()
                    if seccomp_mode != "0":
                        info["restrictions"].append(f"Seccomp mode: {seccomp_mode}")
                    break
    except (FileNotFoundError, PermissionError):
        pass

    # Check capabilities
    success, caps = run_cmd(["cat", "/proc/self/status"])
    if success:
        for line in caps.split("\n"):
            if line.startswith("Cap"):
                name, value = line.split(":", 1)
                if value.strip() == "0000000000000000":
                    info["restrictions"].append(f"{name.strip()} is empty")

    # Check for read-only filesystems
    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    mount_point = parts[1]
                    options = parts[3]
                    if "ro" in options.split(",") and mount_point in ["/", "/usr", "/bin"]:
                        info["restrictions"].append(f"Read-only mount: {mount_point}")
    except (FileNotFoundError, PermissionError):
        pass

    return info

def check_network() -> Dict[str, Any]:
    """Check network capabilities."""
    info = {
        "hostname": socket.gethostname(),
        "can_resolve_dns": False,
        "can_reach_internet": False,
        "interfaces": [],
        "listening_ports": [],
    }

    # Check DNS resolution
    try:
        socket.gethostbyname("google.com")
        info["can_resolve_dns"] = True
    except socket.gaierror:
        pass

    # Try to reach internet (without actually connecting)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(("8.8.8.8", 53))
        info["can_reach_internet"] = result == 0
        sock.close()
    except (socket.error, OSError):
        pass

    # Get network interfaces
    success, ip_output = run_cmd(["ip", "addr"])
    if success:
        current_iface = None
        for line in ip_output.split("\n"):
            if line and not line.startswith(" "):
                parts = line.split(":")
                if len(parts) >= 2:
                    current_iface = parts[1].strip()
            elif "inet " in line and current_iface:
                ip = line.strip().split()[1]
                info["interfaces"].append({"name": current_iface, "ip": ip})

    # Check listening ports
    success, ss_output = run_cmd(["ss", "-tlnp"])
    if success:
        for line in ss_output.split("\n")[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 4:
                info["listening_ports"].append(parts[3])

    return info

def check_filesystem() -> Dict[str, Any]:
    """Check filesystem access and permissions."""
    info = {
        "writable_dirs": [],
        "important_paths": {},
        "disk_usage": {},
    }

    # Check common writable directories
    test_dirs = [
        "/tmp",
        "/var/tmp",
        os.path.expanduser("~"),
        "/home",
        "/opt",
        "/usr/local",
        "/root",
    ]

    for d in test_dirs:
        path = Path(d)
        if path.exists():
            readable = os.access(d, os.R_OK)
            writable = os.access(d, os.W_OK)
            executable = os.access(d, os.X_OK)

            perms = []
            if readable: perms.append("r")
            if writable: perms.append("w")
            if executable: perms.append("x")

            info["important_paths"][d] = "".join(perms) if perms else "none"
            if writable:
                info["writable_dirs"].append(d)

    # Get disk usage
    success, df_output = run_cmd(["df", "-h", "/"])
    if success:
        lines = df_output.split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 5:
                info["disk_usage"] = {
                    "total": parts[1],
                    "used": parts[2],
                    "available": parts[3],
                    "percent": parts[4],
                }

    return info

def check_available_tools() -> Dict[str, List[str]]:
    """Check for available development tools and runtimes."""
    tools = {
        "languages": [],
        "package_managers": [],
        "containers": [],
        "version_control": [],
        "networking": [],
        "editors": [],
        "build_tools": [],
        "shells": [],
    }

    tool_categories = {
        "languages": ["python", "python3", "node", "npm", "go", "rust", "cargo", "ruby", "java", "javac", "gcc", "g++", "clang"],
        "package_managers": ["pip", "pip3", "apt", "apt-get", "yum", "dnf", "brew", "conda", "poetry"],
        "containers": ["docker", "podman", "kubectl", "helm", "docker-compose"],
        "version_control": ["git", "hg", "svn"],
        "networking": ["curl", "wget", "ssh", "scp", "rsync", "nc", "nmap", "dig", "nslookup"],
        "editors": ["vim", "vi", "nano", "emacs", "code"],
        "build_tools": ["make", "cmake", "ninja", "meson", "bazel"],
        "shells": ["bash", "zsh", "fish", "sh", "dash"],
    }

    for category, cmds in tool_categories.items():
        for cmd in cmds:
            if check_command_exists(cmd):
                # Get version if possible
                version = ""
                if cmd in ["python", "python3"]:
                    _, version = run_cmd([cmd, "--version"])
                elif cmd in ["node", "npm", "go", "cargo", "ruby"]:
                    _, version = run_cmd([cmd, "--version"])
                elif cmd == "git":
                    _, version = run_cmd([cmd, "--version"])
                elif cmd == "docker":
                    _, version = run_cmd([cmd, "--version"])

                entry = cmd
                if version:
                    # Extract just the version number
                    version = version.split("\n")[0][:50]
                    entry = f"{cmd} ({version})"
                tools[category].append(entry)

    return tools

def check_resource_limits() -> Dict[str, Any]:
    """Check resource limits (ulimits)."""
    limits = {}

    limit_names = {
        resource.RLIMIT_AS: "virtual_memory",
        resource.RLIMIT_CORE: "core_file_size",
        resource.RLIMIT_CPU: "cpu_time",
        resource.RLIMIT_DATA: "data_segment",
        resource.RLIMIT_FSIZE: "file_size",
        resource.RLIMIT_NOFILE: "open_files",
        resource.RLIMIT_NPROC: "processes",
        resource.RLIMIT_STACK: "stack_size",
    }

    for limit_const, name in limit_names.items():
        try:
            soft, hard = resource.getrlimit(limit_const)
            limits[name] = {
                "soft": soft if soft != resource.RLIM_INFINITY else "unlimited",
                "hard": hard if hard != resource.RLIM_INFINITY else "unlimited",
            }
        except (ValueError, OSError):
            pass

    return limits

def check_environment_vars() -> Dict[str, str]:
    """Get relevant environment variables (sanitized)."""
    relevant_vars = [
        "PATH", "HOME", "USER", "SHELL", "TERM", "LANG",
        "PWD", "OLDPWD", "HOSTNAME",
        "PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV",
        "DOCKER_HOST", "KUBERNETES_SERVICE_HOST",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        "SSH_AUTH_SOCK", "SSH_AGENT_PID",
        "DISPLAY", "XDG_RUNTIME_DIR",
        "CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL",
    ]

    env_info = {}
    for var in relevant_vars:
        value = os.environ.get(var)
        if value:
            # Truncate long values
            if len(value) > 200:
                value = value[:200] + "..."
            env_info[var] = value

    return env_info

def print_section(title: str, data: Any, indent: int = 2):
    """Print a section with formatting."""
    print(f"\n{color(title, Colors.BOLD + Colors.CYAN)}")

    if isinstance(data, dict):
        for key, value in data.items():
            prefix = " " * indent
            if isinstance(value, dict):
                print(f"{prefix}{color(key + ':', Colors.BOLD)}")
                for k, v in value.items():
                    print(f"{prefix}  {k}: {v}")
            elif isinstance(value, list):
                if value:
                    print(f"{prefix}{color(key + ':', Colors.BOLD)}")
                    for item in value:
                        print(f"{prefix}  • {item}")
            else:
                print(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        for item in data:
            print(f"  • {item}")

def main():
    parser = argparse.ArgumentParser(description="Analyze the current execution environment")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = {
        "system": get_system_info(),
        "user": get_user_info(),
        "container": detect_container(),
        "sandbox": detect_sandbox(),
        "network": check_network(),
        "filesystem": check_filesystem(),
        "tools": check_available_tools(),
        "resource_limits": check_resource_limits(),
        "environment": check_environment_vars(),
    }

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return

    # Pretty print
    print(f"\n{color('=' * 60, Colors.BOLD)}")
    print(f"{color('🔍 ENVIRONMENT ANALYSIS', Colors.BOLD + Colors.MAGENTA)}")
    print(f"{color('=' * 60, Colors.BOLD)}")

    # System Info
    print_section("📊 System Information", results["system"])

    # User Info
    user = results["user"]
    print_section("👤 User Context", {
        "username": f"{user['username']} (uid={user['uid']}, gid={user['gid']})",
        "home": user["home"],
        "cwd": user["cwd"],
        "is_root": color("YES", Colors.RED) if user["is_root"] else "no",
        "passwordless_sudo": color("YES", Colors.YELLOW) if user["has_passwordless_sudo"] else "no",
        "groups": ", ".join(user["groups"][:5]) + ("..." if len(user["groups"]) > 5 else ""),
    })

    # Container Detection
    container = results["container"]
    if container["is_container"]:
        print_section("📦 Container Detection", {
            "status": color(f"RUNNING IN {container['container_type'].upper()}", Colors.YELLOW),
            "indicators": container["indicators"],
        })
    else:
        print(f"\n{color('📦 Container Detection', Colors.BOLD + Colors.CYAN)}")
        print(f"  Not running in a detected container")

    # Sandbox Detection
    sandbox = results["sandbox"]
    if sandbox["restrictions"] or sandbox["indicators"]:
        print_section("🔒 Sandbox/Restrictions", {
            "indicators": sandbox["indicators"],
            "restrictions": sandbox["restrictions"],
        })
    else:
        print(f"\n{color('🔒 Sandbox/Restrictions', Colors.BOLD + Colors.CYAN)}")
        print(f"  No obvious sandbox restrictions detected")

    # Network
    net = results["network"]
    print_section("🌐 Network", {
        "hostname": net["hostname"],
        "dns_resolution": color("✓ working", Colors.GREEN) if net["can_resolve_dns"] else color("✗ blocked", Colors.RED),
        "internet_access": color("✓ available", Colors.GREEN) if net["can_reach_internet"] else color("✗ blocked", Colors.RED),
        "interfaces": [f"{i['name']}: {i['ip']}" for i in net["interfaces"][:5]],
    })

    # Filesystem
    fs = results["filesystem"]
    print_section("💾 Filesystem", {
        "writable_directories": fs["writable_dirs"][:5],
        "disk_usage": f"{fs['disk_usage'].get('used', '?')}/{fs['disk_usage'].get('total', '?')} ({fs['disk_usage'].get('percent', '?')})" if fs["disk_usage"] else "unknown",
    })

    # Path permissions
    print(f"\n{color('  Path Permissions:', Colors.BOLD)}")
    for path, perms in fs["important_paths"].items():
        perm_color = Colors.GREEN if 'w' in perms else Colors.YELLOW if 'r' in perms else Colors.RED
        print(f"    {path}: {color(perms, perm_color)}")

    # Available Tools
    tools = results["tools"]
    print_section("🛠️  Available Tools", {
        k: v for k, v in tools.items() if v
    })

    # Resource Limits
    limits = results["resource_limits"]
    print(f"\n{color('📈 Resource Limits', Colors.BOLD + Colors.CYAN)}")
    important_limits = ["open_files", "processes", "virtual_memory"]
    for name in important_limits:
        if name in limits:
            soft = limits[name]["soft"]
            hard = limits[name]["hard"]
            print(f"  {name}: {soft} (soft) / {hard} (hard)")

    # Key Environment Variables
    env = results["environment"]
    print(f"\n{color('🔧 Key Environment Variables', Colors.BOLD + Colors.CYAN)}")
    for var in ["PATH", "SHELL", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "CI"]:
        if var in env:
            value = env[var]
            if var == "PATH":
                # Show first few PATH entries
                paths = value.split(":")[:3]
                value = ":".join(paths) + "..." if len(value.split(":")) > 3 else value
            print(f"  {var}={value}")

    # Summary
    print(f"\n{color('=' * 60, Colors.BOLD)}")
    print(f"{color('SUMMARY', Colors.BOLD)}")
    print(f"{color('=' * 60, Colors.BOLD)}")

    warnings = []
    if user["is_root"]:
        warnings.append("⚠️  Running as root - full system access")
    if user["has_passwordless_sudo"]:
        warnings.append("⚠️  Passwordless sudo available")
    if container["is_container"]:
        warnings.append(f"ℹ️  Running in {container['container_type']} container")
    if not net["can_reach_internet"]:
        warnings.append("ℹ️  No internet access detected")
    if sandbox["restrictions"]:
        warnings.append(f"🔒 {len(sandbox['restrictions'])} restriction(s) detected")

    if warnings:
        for w in warnings:
            print(f"  {w}")
    else:
        print(f"  {color('✓ Standard execution environment', Colors.GREEN)}")

    print()

if __name__ == "__main__":
    main()
