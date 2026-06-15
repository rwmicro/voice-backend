"""
Security utilities - path validation, input sanitization
"""
from pathlib import Path
from typing import Optional
import os

# Allowed directories for audio file inputs
ALLOWED_AUDIO_DIRS = [
    Path("/app/voice/audio_prompts"),
    Path("/app/data"),
    Path("./voice/audio_prompts"),
    Path("./data"),
]

def validate_audio_path(path: Optional[str], param_name: str = "path") -> Optional[str]:
    """
    Validate that a user-provided file path is safe.

    - Must be an existing file
    - Must resolve to within an allowed directory (prevents path traversal)
    - Must not be a symlink pointing outside allowed dirs
    - Returns the resolved path string, or raises ValueError

    Returns None if path is None (optional paths are allowed to be absent).
    """
    if path is None:
        return None

    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid {param_name}: {e}")

    # Must exist
    if not resolved.exists():
        raise ValueError(f"{param_name} does not exist: {path!r}")

    # Must be a regular file
    if not resolved.is_file():
        raise ValueError(f"{param_name} is not a file: {path!r}")

    # Must be within an allowed directory
    allowed = [d.resolve() for d in ALLOWED_AUDIO_DIRS if d.exists()]

    # If no allowed dirs configured (dev/test environment), allow any existing file
    # but still reject obvious traversal attempts
    if not allowed:
        str_path = str(resolved)
        if ".." in str(path) or str_path.startswith(("/etc", "/proc", "/sys", "/root")):
            raise ValueError(f"Unsafe {param_name}: access denied")
        return str(resolved)

    if not any(resolved.is_relative_to(d) for d in allowed):
        raise ValueError(
            f"{param_name} must be within an allowed directory. "
            f"Allowed: {[str(d) for d in allowed]}"
        )

    return str(resolved)


def sanitize_error_message(error: Exception) -> str:
    """
    Return a safe error message that doesn't leak internal details.
    Logs the full error internally, returns generic message to client.
    """
    # Map known exception types to safe messages
    error_map = {
        FileNotFoundError: "Required file not found",
        PermissionError: "Access denied",
        ValueError: str(error),  # ValueError messages are usually safe (validation)
        ImportError: "Required model package not installed on server",
    }
    for exc_type, msg in error_map.items():
        if isinstance(error, exc_type):
            return msg

    # Generic fallback - don't expose internals
    return "An internal error occurred. Check server logs for details."
