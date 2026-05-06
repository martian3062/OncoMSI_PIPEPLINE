import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.conf import settings

from .models import VMTarget
from .registry import ensure_default_vm_target, get_default_vm_target


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def default_vm_target() -> VMTarget:
    existing = get_default_vm_target()
    if existing is not None:
        return existing
    return ensure_default_vm_target()


def vm_health() -> dict[str, Any]:
    target = default_vm_target()
    python_check = run_shell(target, f'"{target.runner_python}" --version')
    runner_check = run_shell(target, f'test -f "{settings.VM_RUNNER_SCRIPT}" && echo present')
    return {
        "target": target.name,
        "execution_mode": target.execution_mode,
        "ssh_user": target.ssh_user,
        "ssh_host": target.ssh_host,
        "project_root": target.project_root,
        "runner_python": target.runner_python,
        "runner_script": settings.VM_RUNNER_SCRIPT,
        "python_ok": python_check.returncode == 0,
        "python_version": (python_check.stdout or python_check.stderr).strip(),
        "runner_script_present": "present" in runner_check.stdout,
    }


def run_shell(target: VMTarget, remote_command: str, timeout: int = 120) -> CommandResult:
    if target.execution_mode == "local":
        proc = subprocess.run(
            ["bash", "-lc", remote_command],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    else:
        command = ["ssh"]
        if target.ssh_key_path:
            command.extend(["-i", target.ssh_key_path])
        command.append(f"{target.ssh_user}@{target.ssh_host}")
        command.append(remote_command)
        proc = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    return CommandResult(proc.returncode, proc.stdout, proc.stderr)


def upload_text(target: VMTarget, remote_path: str, content: str) -> None:
    if target.execution_mode == "local":
        path = Path(remote_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as handle:
        handle.write(content)
        temp_path = handle.name
    try:
        run_shell(target, f'mkdir -p "{Path(remote_path).parent.as_posix()}"')
        command = ["scp"]
        if target.ssh_key_path:
            command.extend(["-i", target.ssh_key_path])
        command.extend([temp_path, f"{target.ssh_user}@{target.ssh_host}:{remote_path}"])
        proc = subprocess.run(command, text=True, capture_output=True, timeout=120, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout or f"Failed to copy file to {remote_path}")
    finally:
        Path(temp_path).unlink(missing_ok=True)


def read_json_file(target: VMTarget, remote_path: str) -> dict[str, Any]:
    result = run_shell(target, f'cat "{remote_path}"', timeout=120)
    if result.returncode != 0:
        raise FileNotFoundError(result.stderr or result.stdout or f"Missing remote file: {remote_path}")
    return json.loads(result.stdout)


def glob_latest_json(target: VMTarget, pattern: str, filename: str) -> tuple[str, dict[str, Any]] | None:
    command = (
        f'latest=$(/bin/ls -dt {pattern} 2>/dev/null | head -n 1); '
        f'if [ -n "$latest" ] && [ -f "$latest/{filename}" ]; then '
        f'echo "$latest/{filename}"; cat "$latest/{filename}"; fi'
    )
    result = run_shell(target, command, timeout=120)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    lines = result.stdout.splitlines()
    path = lines[0].strip()
    payload = json.loads("\n".join(lines[1:]))
    return path, payload
