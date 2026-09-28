"""Small CTFd helper library for the workshop.

Credentials are loaded from environment variables (or .env via python-dotenv):

    CTFD_URL=https://learn.em0.at
    CTFD_TOKEN=...            # CTFd -> Settings -> Access Tokens
    CTFD_CATEGORY=pentesting-with-ai

Public functions:
    spawn_challenge(challenge)
    get_challenge_description(challenge)
    submit_flag(challenge, flag)

`challenge` may be a workshop number ("3"), a case-insensitive name
substring ("haystack"), or a raw CTFd id ("-7619548").
"""

import os
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

load_dotenv()


class CTFError(RuntimeError):
    """Raised when a CTFd operation cannot be completed."""


def _base_url() -> str:
    return os.environ.get("CTFD_URL", "https://learn.em0.at").rstrip("/") + "/"


def _category() -> str:
    return os.environ.get("CTFD_CATEGORY", "pentesting-with-ai")


def _login() -> requests.Session:
    """Return a CTFd session authenticated with the participant's API token.

    We use a CTFd API token (Settings -> Access Tokens) rather than the login
    form. The instance route ``/api/yctf/<id>`` is a plugin route, but CTFd's
    token handler is a global ``before_request`` that only fires on
    ``application/json`` requests -- so we send that Content-Type on every
    request, including GETs, and token auth then covers the plugin route too.
    """
    token = os.environ.get("CTFD_TOKEN")
    if not token or token in {"", "put-your-ctfd-token-here"}:
        raise CTFError("Set CTFD_TOKEN in .env first (CTFd -> Settings -> Access Tokens).")

    session = requests.Session()
    session.headers.update(
        {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    )

    try:
        me = session.get(urljoin(_base_url(), "api/v1/users/me"), timeout=20)
    except requests.RequestException as exc:
        raise CTFError(f"Could not connect to CTFd: {exc}") from exc

    if me.status_code == 401:
        raise CTFError("CTFd rejected the token. Check CTFD_TOKEN.")
    try:
        me.raise_for_status()
        if not me.json().get("success"):
            raise CTFError("CTFd token check failed. Check CTFD_TOKEN.")
    except requests.RequestException as exc:
        raise CTFError(f"Could not connect to CTFd: {exc}") from exc
    except ValueError as exc:
        raise CTFError("CTFd returned an unexpected response while checking the token.") from exc

    return session


def _resolve(session: requests.Session, want: str | int) -> dict:
    """Resolve a workshop number, name substring, or raw CTFd id."""
    want = str(want)
    try:
        response = session.get(urljoin(_base_url(), "api/v1/challenges"), timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise CTFError(f"Could not list CTFd challenges: {exc}") from exc
    except ValueError as exc:
        raise CTFError("CTFd returned an invalid challenge list.") from exc

    if not payload.get("success"):
        raise CTFError(f"Could not list CTFd challenges: {payload}")

    challenges = payload["data"]
    category = _category()
    ours = sorted(
        (challenge for challenge in challenges if challenge.get("category") == category),
        key=lambda challenge: (challenge.get("value", 0), challenge.get("id", 0)),
    )

    if want.lstrip("-").isdigit():
        number = int(want)
        if 1 <= number <= len(ours):
            return ours[number - 1]

        hit = next((challenge for challenge in challenges if challenge["id"] == number), None)
        if hit:
            return hit

    if not ours:
        raise CTFError(
            f"No challenges in category {category!r} are visible. "
            "They may still be hidden; ask an organiser or use a raw CTFd id."
        )

    hits = [
        challenge
        for challenge in ours
        if want.lower() in challenge["name"].lower()
    ]

    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise CTFError("Ambiguous challenge: " + ", ".join(ch["name"] for ch in hits))

    known = "\n".join(
        f"  {index}. {challenge['name']}"
        for index, challenge in enumerate(ours, 1)
    )
    raise CTFError(f"No challenge matching {want!r}. Known:\n{known}")


def _instance(session: requests.Session, challenge_id: int, action: str | None = None):
    path = urljoin(_base_url(), f"api/yctf/{challenge_id}")
    if action:
        return session.post(path, json={"action": action}, timeout=60)
    return session.get(path, timeout=30)


def _response_body(response: requests.Response):
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        try:
            return response.json()
        except ValueError:
            pass
    return response.text


def spawn_challenge(challenge: str | int) -> dict:
    """Start (or reuse) a per-user challenge instance.

    Returns its CTFd id, name, connection information, and expiry.
    Offline/standard challenges without an instance raise CTFError.
    """
    session = _login()
    resolved = _resolve(session, challenge)
    challenge_id = resolved["id"]
    name = resolved["name"]

    try:
        response = _instance(session, challenge_id)
        if not response.ok:
            response = _instance(session, challenge_id, "create")
    except requests.RequestException as exc:
        raise CTFError(f"{name}: instance request failed: {exc}") from exc

    if not response.ok:
        body = _response_body(response)
        if resolved.get("type") == "standard":
            raise CTFError(
                f"{name} has no instance. It is an offline challenge; "
                "the provided files are the challenge."
            )
        raise CTFError(f"{name}: could not start challenge instance: {body}")

    try:
        data = response.json()
    except ValueError as exc:
        raise CTFError(f"{name}: instance API returned an invalid response.") from exc

    return {
        "id": challenge_id,
        "name": name,
        "connection_info": data.get("connectInfo", "(no connection info)"),
        "expires": data.get("expires"),
    }


def get_challenge_description(challenge: str | int) -> dict:
    """Return the full visible CTFd description for a challenge."""
    session = _login()
    resolved = _resolve(session, challenge)
    challenge_id = resolved["id"]

    try:
        response = session.get(
            urljoin(_base_url(), f"api/v1/challenges/{challenge_id}"),
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise CTFError(f"Could not load challenge description: {exc}") from exc
    except ValueError as exc:
        raise CTFError("CTFd returned an invalid challenge description.") from exc

    if not payload.get("success"):
        raise CTFError(f"Could not load challenge description: {payload}")

    data = payload["data"]
    return {
        "id": challenge_id,
        "name": data.get("name", resolved["name"]),
        "category": data.get("category"),
        "value": data.get("value"),
        "description": data.get("description", ""),
        "connection_info": data.get("connection_info"),
    }


def submit_flag(challenge: str | int, flag: str) -> dict:
    """Submit a candidate flag to CTFd and return CTFd's verdict."""
    if not flag or not flag.strip():
        raise CTFError("Flag must not be empty.")

    session = _login()
    resolved = _resolve(session, challenge)
    challenge_id = resolved["id"]

    try:
        response = session.post(
            urljoin(_base_url(), "api/v1/challenges/attempt"),
            json={"challenge_id": challenge_id, "submission": flag.strip()},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise CTFError(f"Flag submission failed: {exc}") from exc

    body = _response_body(response)
    if not response.ok:
        raise CTFError(f"Flag submission failed: {body}")
    if not isinstance(body, dict):
        raise CTFError(f"Unexpected CTFd response: {body}")

    data = body.get("data") or {}
    return {
        "id": challenge_id,
        "name": resolved["name"],
        "status": data.get("status"),
        "message": data.get("message"),
        "success": body.get("success", False),
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "spawn_challenge",
            "description": (
                "Start a per-user CTFd challenge instance, or return the existing "
                "instance if it is already running."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "challenge": {
                        "type": "string",
                        "description": (
                            "Challenge selector: workshop number such as '3', "
                            "case-insensitive name substring, or raw CTFd id."
                        ),
                    }
                },
                "required": ["challenge"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_challenge_description",
            "description": "Get the visible CTFd description and metadata for a challenge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "challenge": {
                        "type": "string",
                        "description": (
                            "Challenge selector: workshop number, name substring, "
                            "or raw CTFd id."
                        ),
                    }
                },
                "required": ["challenge"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_flag",
            "description": "Submit a candidate flag for a challenge to CTFd.",
            "parameters": {
                "type": "object",
                "properties": {
                    "challenge": {
                        "type": "string",
                        "description": (
                            "Challenge selector: workshop number, name substring, "
                            "or raw CTFd id."
                        ),
                    },
                    "flag": {
                        "type": "string",
                        "description": "Candidate flag to submit.",
                    },
                },
                "required": ["challenge", "flag"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "spawn_challenge": spawn_challenge,
    "get_challenge_description": get_challenge_description,
    "submit_flag": submit_flag,
}

