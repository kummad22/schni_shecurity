"""Starting harness: give the LLM a challenge and let it drive the CTFd tools.

`hello_llm.py` is a single completion — it never runs the tools it defines.
This is the missing loop: pick a challenge, spawn its instance via CTFd, hand
the model the connection info, and run the tool-calling loop until the flag is
accepted or the step budget runs out.

    python harness.py 3          # by workshop number
    python harness.py haystack   # by name
    python harness.py -7619548   # by raw CTFd id

The CTFd tools (spawn / describe / submit) are wired for you. Reaching *into*
the target — an `http_request` tool, a `run_shell` tool — is the workshop
exercise: add them to ctf.TOOLS / ctf.TOOL_FUNCTIONS and they show up here for
free. See challenges/README.md for what each challenge forces you to build.

Auth is a CTFd API token (CTFD_TOKEN in .env), not a login form.
"""
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

import ctf

MODEL = "qwen/qwen3.8-flash"
MAX_STEPS = 12  # a wrong-turn agent should stop, not bill the workshop key dry

load_dotenv()


def client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "put-your-key-here":
        raise SystemExit(
            "Missing OPENROUTER_API_KEY.\n"
            "Open the .env tab and paste your workshop API key there."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def system_prompt(challenge: str, connection_info: str | None) -> str:
    if connection_info:
        where = (
            f"Your target is a live service:\n  {connection_info}\n"
            "Reach it and recover the flag."
        )
    else:
        where = (
            "This challenge is an offline handout with no live instance — "
            "the provided files are the challenge."
        )
    return (
        "You are an autonomous assistant solving a CTF challenge. "
        f"You are working on challenge {challenge!r}.\n\n"
        f"{where}\n\n"
        "Flags look like CLA{...}. When you have a candidate, submit it with the "
        "submit_flag tool — do not just print it. A rejected flag is normal; keep "
        "going. Stop once submit_flag reports success."
    )


def run_tool(name: str, arguments: str) -> str:
    """Dispatch one model tool call to its ctf.py implementation."""
    func = ctf.TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"error: unknown tool {name!r}"
    try:
        args = json.loads(arguments or "{}")
        return json.dumps(func(**args), default=str)
    except ctf.CTFError as exc:
        return f"error: {exc}"
    except (TypeError, ValueError) as exc:
        return f"error: bad tool arguments: {exc}"


def solve(challenge: str) -> None:
    llm = client()

    # Spawn first so the model starts with a live target to work against.
    try:
        spawned = ctf.spawn_challenge(challenge)
        connection_info = spawned["connection_info"]
        print(f"[harness] spawned {spawned['name']}: {connection_info}")
    except ctf.CTFError as exc:
        connection_info = None  # offline challenge or not spawnable — model is told
        print(f"[harness] no instance: {exc}")

    messages = [
        {"role": "system", "content": system_prompt(challenge, connection_info)},
        {"role": "user", "content": f"Solve challenge {challenge}."},
    ]

    for step in range(1, MAX_STEPS + 1):
        response = llm.chat.completions.create(
            model=MODEL, messages=messages, tools=ctf.TOOLS, max_tokens=800,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            if msg.content:
                print(msg.content)
            print(f"[harness] model stopped after {step} step(s) with no tool call.")
            return

        for call in msg.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            print(f"[harness] {call.function.name}({call.function.arguments}) -> {result}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )
            if call.function.name == "submit_flag" and '"success": true' in result:
                print("[harness] flag accepted \N{PARTY POPPER}")
                return

    print(f"[harness] step budget ({MAX_STEPS}) exhausted without a solve.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    solve(sys.argv[1])
