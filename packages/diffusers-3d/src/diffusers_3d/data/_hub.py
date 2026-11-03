from __future__ import annotations

import huggingface_hub
from huggingface_hub import constants
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError


def require_hub_access(repository: str, *, used_for: str, alternative: str, token: str | bool | None = None) -> None:
    """Raise a ``PermissionError`` that explains how to get access when this account cannot read ``repository``.

    Third-party weights (gated, or under their own license) always load from their original repository; they are never
    converted or redistributed. ``used_for`` names what needs them and ``alternative`` how to supply them instead.
    Offline, or when the Hub cannot be reached, nothing is checked and loading falls back to the cache.
    """

    if constants.HF_HUB_OFFLINE:
        return
    try:
        huggingface_hub.auth_check(repository, token=token)
    except (GatedRepoError, RepositoryNotFoundError) as error:
        raise PermissionError(
            f"{used_for} loads from {repository}, which this account cannot access. Accept its license at "
            f"https://huggingface.co/{repository}, log in with `hf auth login` (or pass token=...), or "
            f"{alternative}."
        ) from error
    except HfHubHTTPError as error:
        response = getattr(error, "response", None)
        if response is not None and response.status_code in (401, 403):
            raise PermissionError(
                f"{used_for} loads from {repository}, which this account cannot access ({response.status_code}). "
                f"Accept its license at https://huggingface.co/{repository}, log in with `hf auth login` (or pass "
                f"token=...), or {alternative}."
            ) from error
    except Exception:  # noqa: BLE001 - unreachable Hub: the cached files may still load, and downloads raise their own
        return


__all__ = ["require_hub_access"]
