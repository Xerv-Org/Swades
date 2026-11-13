# Swades

A single-file agent for coding, plus plugins for anything else. Python 3.10+, standard library only.

Status: **beta, supervised use only.** See "Status and limits".

## Quick start

```bash
export AGENT_PROVIDER=groq
export GROQ_API_KEY=...              # or OPENROUTER_API_KEY with AGENT_PROVIDER=openrouter
export AGENT_MODEL=<model name>
export AGENT_PRICE_IN=<usd per million input tokens>
export AGENT_PRICE_OUT=<usd per million output tokens>
export AGENT_BUDGET=1.00             # hard monthly cap in USD
cd /path/to/your/repo                # the workspace is the current directory
python3 /path/to/main.py --plan "add input validation to signup()"
```

Set `inColab` near the top of `main.py` to `True` on Colab, `False` elsewhere, or use `"google.colab" in sys.modules`. Admin commands work only when it is `False`.

## Modes

| Flag | Behaviour |
|---|---|
| `--plan` | Read-only tools. Returns a plan, changes nothing. |
| `--ask` | Asks before every write or command. Answer y, n, or a (allow all). |
| `--auto` | Default. No prompts. |
| `--json` | One JSON line on stdout (`status`, `exit`, `result`, `session`, `checkpoint`, `ledger`), logs on stderr. |
| `--continue [id]` | Resume the latest unfinished session, or a given one. |

Exit codes: 0 done or plan, 1 error, 2 stalled or limit, 3 halted (budget or kill switch), 130 interrupted.

## How it works

1. A snapshot of the workspace is taken (shadow git repo in `AGENT_HOME`, your own `.git` is untouched).
2. Each cycle: the agent explores, computes the delta between TASK and reality, calls `plan`, executes, and verifies.
3. When it claims completion, an independent critic re-checks with read-only tools. Rejection starts another cycle.
4. It stops on convergence, a repeated unresolved delta, the cycle limit, the budget, or the kill switch.
5. A ledger prints at the end: time, tokens, cost, calls, errors.

## Built-in tools

`map_repo`, `find_files`, `grep`, `analyse_file`, `read_file` (png/jpeg return as images), `create_file`, `patch_file` (indent-aware, rejects edits that break Python or JSON syntax), `delete_file`, `run`, `web_search` (needs `SERPAPI_API_KEY`), `web_fetch`, `state`, `ask`.
`spawn(task, tools, isolate)` starts sub-agents. Several `spawn` calls in one turn run in parallel, each in a private copy of the workspace merged back afterwards. Conflicts are reported, not overwritten.

## Safety features

- **Checkpoints:** `--checkpoints` lists snapshots, `--undo [id]` restores one. Covers workspace files only, not side effects of commands.
- **Budget cap and kill switch:** `AGENT_BUDGET`; `--stop` and `--resume`; `--status` shows spend.
- **Sandbox:** with `bwrap` installed, `run` gets a read-only root, a writable workspace only, and a hidden state folder. `AGENT_SANDBOX=off|auto|bwrap`. `auto` silently falls back to unsandboxed if `bwrap` is missing or blocked; use `bwrap` to make it mandatory. Without the sandbox, `run` has your user's full file access and a command can leave stray processes.
- **Environment scrub:** `run` and git see only names in `AGENT_RUN_ENV` (default `PATH,HOME,LANG,LC_ALL,TERM,TMPDIR,TEMP,SYSTEMROOT,VIRTUAL_ENV`). API keys never reach commands.
- **Network guard:** `web_fetch` blocks private and loopback addresses, including redirects (`AGENT_FETCH_PRIVATE=1` to allow).
- **Path guard:** file tools cannot leave the workspace or touch `AGENT_HOME`.
- **Audit log:** every tool call goes to `AGENT_HOME/audit.jsonl` (rotates at `AGENT_AUDIT_BYTES`).

## Project memory and skills

- `AGENTS.md` in the workspace is added to the prompt (conventions, commands). It cannot override limits.
- `skills/name.md` or `skills/name/SKILL.md` (optional `description:` front matter) are listed to the agent and loaded on demand with the `skill` tool.

## Plugins

Put them in `plugins/` next to `main.py`, as `plugins/name.py` or `plugins/name/__init__.py`. Names are lowercase letters, digits and underscores. `python3 main.py --plugins` shows what loaded and why anything failed. `AGENT_PLUGINS=a,b` loads only those; `none` loads nothing.

**Important limit:** at most `AGENT_PLUGIN_TOOLS` plugin tools load (default **5**) and extras are skipped with a printed notice. Raise it for the plugins you actually want, and don't load everything; models choose worse with many tools.

### Writing a plugin

```python
VERSION = "1.0"

def setup(api):
    def word_count(a):
        path = (api.root / str(a["path"])).resolve()
        if api.root not in path.parents:
            raise api.ToolError("outside workspace")
        return str(len(path.read_text(encoding="utf-8").split()))

    api.tool("word_count", "Count words in a workspace file.", {"path": api.need()}, word_count, safe=True)
    api.prompt("Use word_count to count words in a file.")
```

| API | Purpose |
|---|---|
| `api.tool(name, desc, params, fn, safe=False)` | Add a tool. `fn(args_dict)` returns a string or `api.Img`, raises `api.ToolError` for expected failures. `safe=True` lets the critic and `--plan` use it. |
| `api.prompt(text)` / `api.check(text)` | Guidance for the agent / for the critic. |
| `api.state(fn)` | Lines shown to the agent every cycle. |
| `api.hook("before_tool", fn)` | Return a string to block the call. `fn(info)` gets `tool`, `args`, `depth`, `role`. |
| `api.hook("after_tool", fn)` | Observe results. Errors are logged, never fatal. |
| `api.skill(name, desc, text)` | Ship a skill. |
| `api.on_exit(fn)` | Cleanup when the run ends. |
| Helpers | `api.need/opt(kind)`, `api.get/num/flag(name)`, `api.root`, `api.data` (your private folder), `api.env()` (scrubbed environment), `api.cfg`, `api.Img`, `api.clip`, `api.image_type`. |

Rules: parameter kinds are `string`, `integer`, `number`, `boolean`, `array`. Tool names cannot clash with built-ins, `plan`, `spawn`, `verdict`, `skill`, or another plugin. A plugin that errors on load is skipped and the rest still load. Every plugin tool call is bounded by `AGENT_TOOL_TIMEOUT` (120s). Hooks are advisory: a hook that blocks `delete_file` does not stop `rm` through `run` unless the sandbox is on.

### Checking a plugin really works in the loop

Loading is not proof. Ask a real model to use it and look at the audit log:

```bash
python3 main.py "Use the word_count tool on notes.txt and tell me the number."
grep '"tool": "word_count"' "$AGENT_HOME/audit.jsonl"
```

If the grep prints nothing, the model never called it. Check that it loaded (`--plugins`) and wasn't dropped by the plugin tool cap.

### Marketplace

Set `AGENT_REGISTRY` to a JSON file (`{"plugins": {"name": {"url": "...", "sha256": "...", "version": "1", "desc": "..."}}}`). `--catalog`, `--install NAME`, `--remove NAME`. A download without a matching sha256 is refused. The hash proves a file is unchanged, **not that it is safe**: plugins run arbitrary code. Read before installing.

### Bundled plugins (all lightly tested)

`notes` (persistent memory), `rest` (allow-listed HTTP, needs `REST_HOSTS`), `mail` (draft-only by default, `MAIL_LIVE=1` to send), `browser` (Playwright), `mcp` (MCP servers over stdio, `MCP_SERVERS` or `MCP_CONFIG`), `cua` (xdotool), `hardware` (ESP32-CAM).

## Best practices

1. Commit first and work on a branch. Review `git diff` before keeping anything.
2. Start with `--plan`, then `--ask`. Use `--auto` for small, well-scoped tasks.
3. Always set `AGENT_BUDGET`. Set `AGENT_CRITIC_MODEL` to a cheap model if cost matters.
4. Run with `AGENT_SANDBOX=bwrap`, ideally inside a container as a non-root user.
5. Keep the Swades folder **outside** the workspace. Otherwise the agent can edit `main.py` and your plugins.
6. Keep `AGENT_HOME` outside the workspace.
7. Write an `AGENTS.md` with build, test and style commands. It's the cheapest quality gain.
8. Keep tasks small. Long vague tasks cost more and stall more.
9. Don't point it at a directory with secrets, production credentials or deploy keys.
10. Treat web pages, file contents and plugin output as untrusted: prompt injection is possible, and a compliant model in `--auto` can delete workspace files (use `--undo`).

## Environment variables (most used)

| Variable | Default | Meaning |
|---|---|---|
| `AGENT_PROVIDER` | groq | `groq` or `openrouter`. `AGENT_BASE_URL` overrides the endpoint. |
| `AGENT_MODEL` | required | Model name. |
| `AGENT_BUDGET`, `AGENT_PRICE_IN/OUT` | 0 | Monthly USD cap and per-million-token prices. OpenRouter cost is read from the response if present. |
| `AGENT_ROOT`, `AGENT_HOME` | cwd, `~/.react-agent` | Workspace and state folder. |
| `AGENT_MODE` | auto | `auto`, `ask`, `plan`. |
| `AGENT_SANDBOX` | auto | `off`, `auto`, `bwrap`. `AGENT_SANDBOX_NET=0` removes network inside the sandbox. |
| `AGENT_CRITIC`, `AGENT_CRITIC_MODEL` | 1, main model | Enable the critic, optionally on another model with `AGENT_CRITIC_PRICE_IN/OUT`. |
| `AGENT_CHILD_MODEL` | main model | Model for sub-agents. |
| `AGENT_MAX_CYCLES`, `AGENT_MAX_STEPS` | 8, 80 | Cycle and per-cycle step limits. |
| `AGENT_MAX_DEPTH`, `AGENT_PARALLEL`, `AGENT_CHILD_CYCLES` | 2, 3, 4 | Sub-agent depth, parallel width, cycles per sub-agent. |
| `AGENT_PLUGIN_TOOLS`, `AGENT_PLUGINS`, `AGENT_PLUGIN_DIR` | 5, all, `./plugins` | Plugin controls. |
| `AGENT_TOOL_TIMEOUT`, `AGENT_RUN_TIMEOUT` | 120, 180 | Seconds for plugin tools and `run`. |
| `AGENT_CHECKPOINT`, `AGENT_MEMORY`, `AGENT_COMPACT` | 1 | Snapshots, `AGENTS.md`, context summarising. |

Other limits are all `AGENT_*` variables near the top of `Config`.

## Status and limits

- **Tested:** offline self-test (`python3 main.py --selftest`) on Python 3.10 to 3.14, offline stress tests, a few live runs on one open model.
- **Not tested live:** critic behaviour, parallel sub-agents, compaction, plan/ask modes, plugin tool calls by a model, OpenRouter, vision, containers.
- **Known gaps:** no OS jail without `bwrap`; hooks are bypassable via `run`; plugins are unsandboxed; checkpoints don't undo external side effects; Windows untested; a single large file is hard to audit.
- **Not for:** unattended operation, production systems, or repos containing secrets.
