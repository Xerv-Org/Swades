import ast
import fnmatch
import html
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

inColab = True

COLAB_HOME = Path("/content")
IGNORE = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode", "dist", "build", ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints"})
RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

DEFS = re.compile(
    r"^\s*(?:(?:export|default|pub|public|private|protected|static|async|abstract|final|extern|unsafe|override)\s+)*"
    r"(?:def|class|function\*?|func|fn|struct|enum|trait|impl|interface|type|object|module|namespace)\b[^\n]*"
    r"|^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|\w+\s*=>)[^\n]*"
)
TOP = re.compile(r"^\S[^\n]*[{:]\s*$")
IMPORTS = re.compile(r"^\s*(?:import|from|use|require|include|using|package)\b|^\s*(?:const|let|var)\s+[^=]+=\s*require\(")

CORE = (
    "You are an autonomous agent in a ReAct loop. Every cycle you receive TASK, INDEX (knowledge map) and STATE.\n"
    "1 Explore with tools until you hold the knowledge relevant to TASK. Batch independent tool calls in one turn.\n"
    "2 Compare the real current state with TASK and compute the delta.\n"
    "3 Call plan(delta, steps). Empty steps means TASK is fully satisfied and verified by evidence you gathered this cycle.\n"
    "4 Execute the steps with tools and verify every result with real checks.\n"
    "5 End the cycle with a brief summary and no tool call. The loop then re-indexes, re-checks state and recomputes the delta.\n"
    "Never assert what you have not observed. Be terse. On tool error, adapt and retry differently."
)

CODE = (
    "Domain: software workspace. Locate with map_repo, find_files, grep. Understand with analyse_file, then read_file on exact ranges. "
    "Change with create_file, patch_file, delete_file. Verify with run. "
    "Look up external facts, docs and error messages with web_search, then read the pages with web_fetch. "
    "patch_file takes either a unique old block or start and end lines (inclusive, end=start-1 inserts before start) plus the replacement new text. "
    "Indentation is re-fitted automatically, but keep relative indentation correct, especially in Python, YAML and Makefiles. "
    "Write complete working code: no placeholders, no TODO markers, no hardcoded secrets or paths; derive values from config, arguments or environment. "
    "Run tests, or at minimum a syntax and import check, after every change set."
)


class AgentError(Exception):
    pass


class ToolError(Exception):
    pass


def _get(name, default=None):
    if inColab:
        try:
            from google.colab import userdata
            value = userdata.get(name)
            if value:
                return value
        except Exception:
            pass
    return os.environ.get(name, default)


def _num(name, default):
    raw = _get(name)
    return type(default)(raw) if raw else default


def clip(text, limit):
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n[{len(text) - limit} chars cut]\n{text[-tail:]}"


@dataclass
class Config:
    key: str
    model: str
    base_url: str
    serp_key: str
    ua: str
    root: Path
    ignore: frozenset
    max_tokens: int
    timeout: int
    retries: int
    max_cycles: int
    max_steps: int
    max_nudges: int
    stall: int
    ctx_chars: int
    keep: int
    out_chars: int
    read_lines: int
    max_matches: int
    max_files: int
    max_bytes: int
    map_depth: int
    map_lines: int
    run_timeout: int
    fetch_chars: int
    search_results: int
    indent: int
    max_tools: int

    @classmethod
    def load(cls, llm=True, root=None):
        key = _get("GROQ_API_KEY") or ""
        model = _get("AGENT_MODEL") or ""
        if llm:
            missing = [n for n, v in (("GROQ_API_KEY", key), ("AGENT_MODEL", model)) if not v]
            if missing:
                where = "Colab Secrets (key icon, notebook access on)" if inColab else "environment variables"
                raise SystemExit(f"missing {', '.join(missing)} in {where}")
        chosen = root or _get("AGENT_ROOT") or (COLAB_HOME / "workspace" if inColab else Path.cwd())
        base = Path(chosen).resolve()
        base.mkdir(parents=True, exist_ok=True)
        extra = frozenset(filter(None, (_get("AGENT_IGNORE") or "").split(",")))
        return cls(
            key=key,
            model=model,
            base_url=(_get("AGENT_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/"),
            serp_key=_get("SERPAPI_API_KEY") or "",
            ua=_get("AGENT_UA") or "react-agent",
            root=base,
            ignore=IGNORE | extra,
            max_tokens=_num("AGENT_MAX_TOKENS", 8192),
            timeout=_num("AGENT_TIMEOUT", 180),
            retries=_num("AGENT_RETRIES", 5),
            max_cycles=_num("AGENT_MAX_CYCLES", 8),
            max_steps=_num("AGENT_MAX_STEPS", 80),
            max_nudges=_num("AGENT_MAX_NUDGES", 3),
            stall=_num("AGENT_STALL", 2),
            ctx_chars=_num("AGENT_CTX_CHARS", 400000),
            keep=_num("AGENT_KEEP", 10),
            out_chars=_num("AGENT_OUT_CHARS", 8000),
            read_lines=_num("AGENT_READ_LINES", 400),
            max_matches=_num("AGENT_MAX_MATCHES", 80),
            max_files=_num("AGENT_MAX_FILES", 20000),
            max_bytes=_num("AGENT_MAX_BYTES", 2000000),
            map_depth=_num("AGENT_MAP_DEPTH", 3),
            map_lines=_num("AGENT_MAP_LINES", 300),
            run_timeout=_num("AGENT_RUN_TIMEOUT", 180),
            fetch_chars=_num("AGENT_FETCH_CHARS", 10000),
            search_results=_num("AGENT_SEARCH_RESULTS", 8),
            indent=_num("AGENT_INDENT", 4),
            max_tools=_num("AGENT_MAX_TOOLS", 15),
        )


def need(kind="string"):
    return (kind, True)


def opt(kind="string"):
    return (kind, False)


@dataclass(frozen=True)
class Tool:
    name: str
    desc: str
    params: dict
    fn: Callable

    def schema(self):
        props, required = {}, []
        for key, (kind, mandatory) in self.params.items():
            prop = {"type": kind}
            if kind == "array":
                prop["items"] = {"type": "string"}
            props[key] = prop
            if mandatory:
                required.append(key)
        return {"name": self.name, "description": self.desc, "input_schema": {"type": "object", "properties": props, "required": required}}


@dataclass
class Domain:
    prompt: str
    tools: dict
    index: Callable
    state: Callable


class LLM:
    """Groq client (OpenAI-compatible chat completions).

    The agent loop keeps using block-style messages internally; this class converts
    them to Groq's wire format and converts the reply back, so nothing else changes.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.url = f"{cfg.base_url}/chat/completions"
        self.headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {cfg.key}",
            "user-agent": cfg.ua,
        }

    @staticmethod
    def _to_wire_tools(tools):
        return [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools
        ]

    @staticmethod
    def _to_wire_messages(system, messages):
        text = system if isinstance(system, str) else "\n".join(b["text"] for b in system if b.get("type") == "text")
        out = [{"role": "system", "content": text}]
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                out.append({"role": message["role"], "content": content})
                continue
            if message["role"] == "assistant":
                said = "".join(b["text"] for b in content if b.get("type") == "text")
                calls = [
                    {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}}
                    for b in content
                    if b.get("type") == "tool_use"
                ]
                entry = {"role": "assistant", "content": said or None}
                if calls:
                    entry["tool_calls"] = calls
                out.append(entry)
                continue
            extra = []
            for b in content:
                if b.get("type") == "tool_result":
                    body = b["content"] if isinstance(b["content"], str) else json.dumps(b["content"], ensure_ascii=False)
                    if b.get("is_error"):
                        body = "ERROR: " + body
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": body})
                elif b.get("type") == "text" and b.get("text"):
                    extra.append(b["text"])
            if extra:
                out.append({"role": "user", "content": "\n".join(extra)})
        return out

    @staticmethod
    def _from_wire(data):
        choices = data.get("choices") or []
        if not choices:
            raise AgentError(f"api returned no choices: {clip(json.dumps(data), 500)}")
        choice = choices[0]
        message = choice.get("message") or {}
        blocks = []
        if message.get("content"):
            blocks.append({"type": "text", "text": message["content"]})
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            blocks.append({"type": "tool_use", "id": call["id"], "name": fn.get("name", ""), "input": args})
        finish = choice.get("finish_reason")
        if finish == "length":
            stop = "max_tokens"
        elif any(b["type"] == "tool_use" for b in blocks):
            stop = "tool_use"
        else:
            stop = "end_turn"
        return {"content": blocks, "stop_reason": stop}

    def send(self, system, tools, messages):
        payload = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "messages": self._to_wire_messages(system, messages),
            "tools": self._to_wire_tools(tools),
            "tool_choice": "auto",
        }
        body = json.dumps(payload).encode("utf-8")
        delay = 1.0
        for attempt in range(self.cfg.retries + 1):
            request = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
            wait = delay
            try:
                with urllib.request.urlopen(request, timeout=self.cfg.timeout) as response:
                    return self._from_wire(json.load(response))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                transient = exc.code in RETRYABLE or (exc.code == 400 and "tool_use_failed" in detail)
                if not transient or attempt == self.cfg.retries:
                    raise AgentError(f"api {exc.code}: {clip(detail, 800)}")
                try:
                    wait = float(exc.headers.get("retry-after") or delay)
                except ValueError:
                    wait = delay
            except OSError as exc:
                if attempt == self.cfg.retries:
                    raise AgentError(f"network: {exc}")
            time.sleep(min(wait, 60) + random.uniform(0, 0.5))
            delay = min(delay * 2, 30)
        raise AgentError("retries exhausted")


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.root
        self.seen = self._snapshot()

    def _path(self, raw):
        target = (self.root / raw).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"outside workspace: {raw}")
        return target

    def _rel(self, path):
        return path.relative_to(self.root).as_posix()

    def _walk(self, base):
        count = 0
        for dirpath, dirs, names in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in self.cfg.ignore)
            for name in sorted(names):
                if name in self.cfg.ignore:
                    continue
                full = Path(dirpath) / name
                try:
                    stat = full.stat()
                except OSError:
                    continue
                yield full, stat
                count += 1
                if count >= self.cfg.max_files:
                    return

    def _paths(self, base):
        if base.is_file():
            yield base
        else:
            for full, _ in self._walk(base):
                yield full

    def _snapshot(self):
        return {self._rel(f): (s.st_mtime_ns, s.st_size) for f, s in self._walk(self.root)}

    def _read(self, path):
        if not path.is_file():
            raise ToolError(f"not a file: {path.name}")
        data = path.read_bytes()
        if b"\0" in data[:2048]:
            raise ToolError("binary file")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError("not utf-8")

    def _peek(self, path):
        try:
            if path.stat().st_size > self.cfg.max_bytes:
                return None
            data = path.read_bytes()
        except OSError:
            return None
        if b"\0" in data[:2048]:
            return None
        return data.decode("utf-8", "replace").replace("\r\n", "\n")

    def _write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".agent.tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        if path.exists():
            shutil.copymode(path, tmp)
        os.replace(tmp, path)

    def _check(self, path, text):
        suffix = path.suffix.lower()
        try:
            if suffix == ".py":
                ast.parse(text)
            elif suffix in (".json", ".ipynb"):
                json.loads(text)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return ""

    @staticmethod
    def _split(text):
        nl = "\r\n" if "\r\n" in text else "\n"
        return text.replace("\r\n", "\n").split("\n"), nl

    @staticmethod
    def _count(lines):
        return len(lines) - (1 if lines[-1] == "" else 0)

    @staticmethod
    def _size(n):
        if n < 1024:
            return f"{n}b"
        if n < 1048576:
            return f"{n / 1024:.1f}k"
        return f"{n / 1048576:.1f}m"

    @staticmethod
    def _indent(line):
        return line[: len(line) - len(line.lstrip())]

    def _leaves(self, node):
        return sum(self._leaves(v) if isinstance(v, dict) else 1 for v in node.values())

    def _render(self, node, level, depth, out):
        for name in sorted(node, key=lambda k: (not k.endswith("/"), k)):
            child = node[name]
            pad = " " * level
            if isinstance(child, dict):
                out.append(f"{pad}{name} {self._leaves(child)}")
                if level < depth:
                    self._render(child, level + 1, depth, out)
            else:
                out.append(f"{pad}{name} {self._size(child)}")

    def map_repo(self, a):
        base = self._path(a.get("path", "."))
        if not base.is_dir():
            raise ToolError("not a directory")
        depth = int(a.get("depth") or self.cfg.map_depth)
        tree, langs, total = {}, Counter(), 0
        for full, stat in self._walk(base):
            parts = full.relative_to(base).parts
            node = tree
            for part in parts[:-1]:
                node = node.setdefault(part + "/", {})
            node[parts[-1]] = stat.st_size
            langs[full.suffix.lower() or full.name] += 1
            total += 1
        lines = []
        self._render(tree, 0, depth, lines)
        if len(lines) > self.cfg.map_lines:
            extra = len(lines) - self.cfg.map_lines
            lines = lines[: self.cfg.map_lines] + [f"[+{extra} lines; narrow path or lower depth]"]
        mix = " ".join(f"{k}:{v}" for k, v in langs.most_common(6))
        cut = " scan-truncated" if total >= self.cfg.max_files else ""
        label = self._rel(base) if base != self.root else "."
        return "\n".join([f"{label} files={total}{cut} {mix}".strip(), *lines])

    def find_files(self, a):
        pattern = a["pattern"]
        variants = {pattern, pattern.replace("**/", "")}
        hits = []
        for full, _ in self._walk(self._path(a.get("path", "."))):
            rel = self._rel(full)
            if any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(full.name, p) for p in variants):
                hits.append(rel)
                if len(hits) >= self.cfg.max_matches:
                    break
        return "\n".join(hits) or "none"

    def grep(self, a):
        flags = re.IGNORECASE if a.get("ignore_case") else 0
        try:
            rx = re.compile(a["pattern"], flags)
        except re.error as exc:
            raise ToolError(f"bad regex: {exc}")
        base = self._path(a.get("path", "."))
        glob = a.get("glob")
        ctx = max(0, int(a.get("context") or 0))
        limit = self.cfg.max_matches
        out, found = [], 0
        for full in self._paths(base):
            rel = self._rel(full)
            if glob and not (fnmatch.fnmatch(full.name, glob) or fnmatch.fnmatch(rel, glob)):
                continue
            text = self._peek(full)
            if text is None:
                continue
            lines = text.split("\n")
            shown = set()
            for i, line in enumerate(lines):
                if not rx.search(line):
                    continue
                for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1)):
                    if j not in shown:
                        shown.add(j)
                        mark = ":" if j == i else "-"
                        out.append(f"{rel}{mark}{j + 1}{mark}{lines[j][:240]}")
                found += 1
                if found >= limit:
                    break
            if found >= limit:
                break
        if not out:
            return "none"
        return "\n".join(out) + ("\n[limit reached]" if found >= limit else "")

    def _py_outline(self, text):
        tree = ast.parse(text)
        imports, out = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append("." * node.level + (node.module or ""))

        def visit(body, level):
            for node in body:
                pad = "  " * level
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                    out.append(f"{pad}{node.lineno}-{node.end_lineno} {kw} {node.name}({ast.unparse(node.args)}){ret}")
                elif isinstance(node, ast.ClassDef):
                    bases = ", ".join(ast.unparse(b) for b in node.bases)
                    out.append(f"{pad}{node.lineno}-{node.end_lineno} class {node.name}({bases})")
                    visit(node.body, level + 1)
                elif level == 0 and isinstance(node, (ast.Assign, ast.AnnAssign)):
                    out.append(f"{node.lineno} {ast.unparse(node).splitlines()[0][:100]}")

        visit(tree.body, 0)
        return sorted(set(imports)), out

    def analyse_file(self, a):
        path = self._path(a["path"])
        text = self._read(path)
        lines, _ = self._split(text)
        suffix = path.suffix.lower()
        head = f"{self._rel(path)} {suffix.lstrip('.') or 'txt'} {self._count(lines)}L {self._size(len(text))}"
        if suffix == ".py":
            try:
                imports, defs = self._py_outline(text)
                return "\n".join([head, "imports: " + ", ".join(imports), *defs[: self.cfg.map_lines]])
            except SyntaxError as exc:
                head += f" syntax-error L{exc.lineno}: {exc.msg}"
        elif suffix == ".json":
            try:
                data = json.loads(text)
            except ValueError:
                head += " invalid-json"
            else:
                if isinstance(data, dict):
                    shape = "keys: " + ", ".join(list(data)[:40])
                elif isinstance(data, list):
                    shape = f"list[{len(data)}]"
                else:
                    shape = type(data).__name__
                return f"{head}\n{shape}"
        imports = [l.strip()[:120] for l in lines if IMPORTS.match(l)][:20]
        defs = [f"{i} {l.rstrip()[:120]}" for i, l in enumerate(lines, 1) if DEFS.match(l)]
        if not defs:
            defs = [f"{i} {l.rstrip()[:120]}" for i, l in enumerate(lines, 1) if TOP.match(l)]
        return "\n".join([head, "imports: " + " | ".join(imports), *defs[: self.cfg.map_lines]])

    def read_file(self, a):
        path = self._path(a["path"])
        lines, _ = self._split(self._read(path))
        total = self._count(lines)
        rel = self._rel(path)
        if total == 0:
            return f"{rel} empty"
        start = max(1, int(a.get("start") or 1))
        if start > total:
            return f"{rel} has {total} lines"
        cap = start + self.cfg.read_lines - 1
        end = min(int(a.get("end") or cap), cap, total)
        body = "\n".join(f"{i}|{lines[i - 1]}" for i in range(start, end + 1))
        return f"{rel} {start}-{end}/{total}\n" + clip(body, self.cfg.out_chars)

    def create_file(self, a):
        path = self._path(a["path"])
        if path.is_dir():
            raise ToolError("is a directory")
        existed = path.exists()
        if existed and not a.get("overwrite"):
            raise ToolError("exists; set overwrite or use patch_file")
        content = a["content"]
        if content and not content.endswith("\n"):
            content += "\n"
        error = self._check(path, content)
        if error:
            raise ToolError(f"syntax: {error}")
        self._write(path, content)
        lines, _ = self._split(content)
        return f"{'overwrote' if existed else 'created'} {self._rel(path)} {self._count(lines)}L"

    def delete_file(self, a):
        path = self._path(a["path"])
        if path == self.root:
            raise ToolError("refuse to delete root")
        if path.is_dir() and not path.is_symlink():
            if a.get("recursive"):
                shutil.rmtree(path)
            else:
                path.rmdir()
        elif path.exists() or path.is_symlink():
            path.unlink()
        else:
            raise ToolError("not found")
        return f"deleted {a['path']}"

    def _locate(self, lines, count, old):
        old_lines = old.replace("\r\n", "\n").split("\n")
        while old_lines and not old_lines[0].strip():
            old_lines.pop(0)
        while old_lines and not old_lines[-1].strip():
            old_lines.pop()
        if not old_lines:
            raise ToolError("old is blank")
        key = [l.strip() for l in old_lines]
        flat = [l.strip() for l in lines[:count]]
        n = len(key)
        hits = [i for i in range(count - n + 1) if flat[i] == key[0] and flat[i : i + n] == key]
        if len(hits) > 1:
            literal = [l.rstrip() for l in old_lines]
            exact = [i for i in hits if [l.rstrip() for l in lines[i : i + n]] == literal]
            if len(exact) == 1:
                hits = exact
        if len(hits) > 1:
            where = ", ".join(str(i + 1) for i in hits[:8])
            raise ToolError(f"ambiguous: matches at lines {where}; add context or use start/end")
        if not hits:
            near = [str(i + 1) for i in range(count) if flat[i] == key[0]][:5]
            hint = f"first line of old seen at L{', L'.join(near)} but the block differs; re-read" if near else "first line of old not found; grep or re-read"
            raise ToolError(f"no match: {hint}")
        return hits[0], hits[0] + n, old_lines

    def _unit(self, lines):
        prev, steps = 0, Counter()
        for line in lines:
            if not line.strip():
                continue
            ws = self._indent(line)
            if ws.startswith("\t"):
                return "\t"
            if len(ws) > prev:
                steps[len(ws) - prev] += 1
            prev = len(ws)
        if not steps:
            return None
        return " " * steps.most_common(1)[0][0]

    def _level(self, ws, unit):
        tabs = ws.count("\t")
        spaces = len(ws) - tabs
        if unit == "\t":
            return tabs + spaces // self.cfg.indent
        return (tabs * self.cfg.indent + spaces) // len(unit)

    def _fit(self, new, basis, anchor, agent_lines, file_lines):
        fu = self._unit(file_lines)
        au = self._unit(agent_lines)
        fu = fu or au or " " * self.cfg.indent
        au = au or fu
        base = self._level(self._indent(basis), au)
        origin = self._level(self._indent(anchor), fu)
        shift = origin - base
        out = []
        if au == fu:
            if shift == 0:
                return new, False
            cut = fu * (-shift)
            for line in new:
                if not line.strip():
                    out.append("")
                elif shift > 0:
                    out.append(fu * shift + line)
                else:
                    out.append(line[len(cut):] if line.startswith(cut) else line.lstrip())
            return out, True
        for line in new:
            if not line.strip():
                out.append("")
            else:
                level = max(0, origin + self._level(self._indent(line), au) - base)
                out.append(fu * level + line.lstrip())
        return out, True

    def patch_file(self, a):
        path = self._path(a["path"])
        text = self._read(path)
        lines, nl = self._split(text)
        count = self._count(lines)
        raw = a["new"].replace("\r\n", "\n")
        new = raw.split("\n") if raw else []
        if raw.endswith("\n"):
            new.pop()
        old_lines = []
        if a.get("old"):
            i, j, old_lines = self._locate(lines, count, a["old"])
        else:
            if a.get("start") is None or a.get("end") is None:
                raise ToolError("give old, or start and end")
            start, end = int(a["start"]), int(a["end"])
            if not (1 <= start <= count + 1 and start - 1 <= end <= count):
                raise ToolError(f"range invalid; file has {count} lines")
            i, j = start - 1, end
        fitted, moved = new, False
        probe = next((l for l in new if l.strip()), None)
        if probe is not None and i < j:
            anchor = next((l for l in lines[i:j] if l.strip()), None)
            if anchor is not None:
                basis = next((l for l in old_lines if l.strip()), probe)
                fitted, moved = self._fit(new, basis, anchor, new + old_lines, lines)
        updated = lines[:i] + fitted + lines[j:]
        if updated == lines:
            return "no change"
        out = nl.join(updated)
        before, after = self._check(path, text), self._check(path, out)
        if after and not before:
            raise ToolError(f"rejected, would break syntax: {after}")
        self._write(path, out)
        note = " reindented" if moved else ""
        return f"patched {self._rel(path)} L{i + 1}-{i + len(fitted)} {count}->{self._count(updated)}L{note}"

    def _kill(self, proc):
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass

    def run(self, a):
        timeout = int(a.get("timeout") or self.cfg.run_timeout)
        proc = subprocess.Popen(
            a["cmd"], shell=True, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, errors="replace", start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            self._kill(proc)
            out, _ = proc.communicate()
            code = f"timeout {timeout}s"
        return f"exit {code}\n" + clip((out or "").strip(), self.cfg.out_chars)

    def web_fetch(self, a):
        url = a["url"]
        if not re.match(r"https?://", url):
            raise ToolError("http(s) only")
        request = urllib.request.Request(url, headers={"User-Agent": self.cfg.ua})
        with urllib.request.urlopen(request, timeout=self.cfg.timeout) as response:
            data = response.read(self.cfg.fetch_chars * 8)
            charset = response.headers.get_content_charset() or "utf-8"
            kind = response.headers.get_content_type()
        text = data.decode(charset, "replace")
        if "html" in kind:
            text = re.sub(r"(?is)<(script|style|noscript|svg)\b.*?</\1>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            text = html.unescape(text)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        return clip(text.strip(), self.cfg.fetch_chars)

    @staticmethod
    def _format_serp(data, limit):
        if data.get("error"):
            raise ToolError(f"search: {data['error']}")
        parts = []
        box = data.get("answer_box") or {}
        answer = box.get("answer") or box.get("result") or box.get("snippet")
        if isinstance(answer, list):
            answer = " ".join(str(x) for x in answer)
        if answer:
            parts.append(f"answer: {answer}")
        graph = data.get("knowledge_graph") or {}
        about = graph.get("description")
        if about:
            title = graph.get("title") or ""
            parts.append(f"about {title}: {about}".strip())
        for n, item in enumerate((data.get("organic_results") or [])[:limit], 1):
            title = item.get("title") or ""
            link = item.get("link") or ""
            snippet = item.get("snippet") or ""
            parts.append(f"{n}. {title}\n   {link}\n   {snippet}".rstrip())
        return "\n".join(parts) or "none"

    def web_search(self, a):
        if not self.cfg.serp_key:
            where = "Colab Secrets (key icon, notebook access on)" if inColab else "environment variables"
            raise ToolError(f"SERPAPI_API_KEY missing in {where}")
        limit = max(1, min(int(a.get("num") or self.cfg.search_results), 20))
        query = urllib.parse.urlencode({"engine": "google", "q": a["query"], "num": limit, "api_key": self.cfg.serp_key})
        request = urllib.request.Request(f"https://serpapi.com/search.json?{query}", headers={"User-Agent": self.cfg.ua})
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise ToolError(f"search http {exc.code}: {clip(detail, 400)}")
        return self._format_serp(data, limit)

    def _git(self):
        if not shutil.which("git"):
            return "git: unavailable"
        try:
            done = subprocess.run(["git", "status", "-sb"], cwd=self.root, capture_output=True, text=True, errors="replace", timeout=self.cfg.run_timeout)
        except subprocess.SubprocessError:
            return "git: error"
        if done.returncode:
            return "git: not a repository"
        return "git: " + clip(done.stdout.strip(), self.cfg.out_chars // 4)

    def state(self, _):
        now = self._snapshot()
        before, self.seen = self.seen, now
        added = sorted(now.keys() - before.keys())
        gone = sorted(before.keys() - now.keys())
        changed = sorted(k for k in now.keys() & before.keys() if now[k] != before[k])
        cap = self.cfg.max_matches
        parts = [f"files={len(now)}"]
        for mark, group in (("+", added), ("~", changed), ("-", gone)):
            if group:
                extra = f" (+{len(group) - cap} more)" if len(group) > cap else ""
                parts.append(f"{mark} {' '.join(group[:cap])}{extra}")
        if len(parts) == 1:
            parts.append("no file changes since last check")
        parts.append(self._git())
        return "\n".join(parts)

    def ask(self, a):
        try:
            return input(f"\n? {a['question']}\n> ").strip() or "(no answer)"
        except EOFError:
            return "no input channel; decide autonomously"

    def domain(self):
        specs = (
            ("map_repo", "Tree with sizes, dir file counts, language mix.", {"path": opt(), "depth": opt("integer")}, self.map_repo),
            ("find_files", "Glob file paths.", {"pattern": need(), "path": opt()}, self.find_files),
            ("grep", "Regex search, output file:line:text. glob filters files.", {"pattern": need(), "path": opt(), "glob": opt(), "context": opt("integer"), "ignore_case": opt("boolean")}, self.grep),
            ("analyse_file", "Outline: imports, classes, functions with line ranges.", {"path": need()}, self.analyse_file),
            ("read_file", "Numbered lines n|text. Range via start,end.", {"path": need(), "start": opt("integer"), "end": opt("integer")}, self.read_file),
            ("create_file", "New file, syntax-checked. Fails if exists unless overwrite.", {"path": need(), "content": need(), "overwrite": opt("boolean")}, self.create_file),
            ("patch_file", "Replace a block. Use old (unique, whitespace-insensitive match) or start+end lines (inclusive; end=start-1 inserts before start). new is the replacement; empty new deletes.", {"path": need(), "new": need(), "old": opt(), "start": opt("integer"), "end": opt("integer")}, self.patch_file),
            ("delete_file", "Delete file or directory.", {"path": need(), "recursive": opt("boolean")}, self.delete_file),
            ("run", "Shell command at workspace root. Returns exit code and merged output.", {"cmd": need(), "timeout": opt("integer")}, self.run),
            ("web_search", "Live Google search via SerpAPI. Returns answer box, knowledge summary and ranked results (title, url, snippet).", {"query": need(), "num": opt("integer")}, self.web_search),
            ("web_fetch", "Fetch URL as plain text.", {"url": need()}, self.web_fetch),
            ("state", "File changes since last check plus git status.", {}, self.state),
            ("ask", "Ask the user one question.", {"question": need()}, self.ask),
        )
        tools = {name: Tool(name, desc, params, fn) for name, desc, params, fn in specs}
        return Domain(CODE, tools, lambda: self.map_repo({}), lambda: self.state({}))


class ReAct:
    def __init__(self, cfg, llm, domain):
        self.cfg = cfg
        self.llm = llm
        self.domain = domain
        self.plan = {"delta": "", "steps": []}
        plan_tool = Tool(
            "plan",
            "Declare the delta between TASK and current state, and ordered steps. Empty steps = task satisfied and verified.",
            {"delta": need(), "steps": need("array")},
            self._plan,
        )
        if "plan" in domain.tools:
            raise AgentError("domain may not define plan")
        self.tools = {"plan": plan_tool, **domain.tools}
        if len(self.tools) > cfg.max_tools:
            raise AgentError(f"{len(self.tools)} tools exceeds limit {cfg.max_tools}")
        self.schemas = [t.schema() for t in self.tools.values()]
        self.system = [{"type": "text", "text": CORE + "\n" + domain.prompt}]

    def _plan(self, a):
        raw = a.get("steps") or []
        steps = [str(s) for s in ([raw] if isinstance(raw, str) else raw)]
        self.plan = {"delta": str(a["delta"]), "steps": steps}
        print(f"  delta: {self.plan['delta']}")
        for n, step in enumerate(steps, 1):
            print(f"  {n}. {step}")
        if not steps:
            return "converged"
        return f"accepted {len(steps)} steps; execute now, then reply with a short summary and no tool call"

    def _call(self, block):
        tool = self.tools.get(block["name"])
        if tool is None:
            return f"unknown tool {block['name']}", True
        try:
            return clip(str(tool.fn(block["input"])), self.cfg.out_chars), False
        except ToolError as exc:
            return str(exc), True
        except KeyError as exc:
            return f"missing argument {exc}", True
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}", True

    def _log(self, cycle, call, out, err):
        args = json.dumps(call["input"], ensure_ascii=False)[:90]
        first = out.split("\n", 1)[0][:100]
        print(f"[{cycle}] {call['name']} {args} -> {'ERR ' if err else ''}{first}", flush=True)

    def _trim(self, msgs):
        if len(json.dumps(msgs)) <= self.cfg.ctx_chars:
            return
        for message in msgs[: -self.cfg.keep]:
            if message["role"] == "user" and isinstance(message["content"], list):
                for block in message["content"]:
                    if block.get("type") == "tool_result" and len(block["content"]) > 200:
                        block["content"] = block["content"][:160] + "\n[trimmed]"

    @staticmethod
    def _clean(blocks):
        out = []
        for b in blocks:
            if b.get("type") == "text" and b.get("text"):
                out.append({"type": "text", "text": b["text"]})
            elif b.get("type") == "tool_use":
                out.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b.get("input") or {}})
        return out

    def _brief(self, task, cycle, carry):
        parts = [
            f"TASK\n{task}",
            f"CYCLE {cycle}/{self.cfg.max_cycles}",
            f"INDEX\n{self.domain.index()}",
            f"STATE\n{self.domain.state()}",
        ]
        if carry:
            parts.append(f"PREVIOUS\n{carry}")
        return "\n\n".join(parts)

    def _cycle(self, msgs, cycle):
        self.plan = {"delta": "", "steps": []}
        planned, used, nudges = False, 0, 0
        for _ in range(self.cfg.max_steps):
            self._trim(msgs)
            reply = self.llm.send(self.system, self.schemas, msgs)
            content = self._clean(reply.get("content", []))
            if reply.get("stop_reason") == "max_tokens":
                kept = [b for b in content if b["type"] == "text"] or [{"type": "text", "text": "(truncated)"}]
                msgs.append({"role": "assistant", "content": kept})
                msgs.append({"role": "user", "content": "Output hit the token limit. Retry with smaller edits."})
                continue
            if not content:
                content = [{"type": "text", "text": "(empty)"}]
            msgs.append({"role": "assistant", "content": content})
            text = "".join(b["text"] for b in content if b["type"] == "text").strip()
            calls = [b for b in content if b["type"] == "tool_use"]
            if not calls:
                if planned:
                    return False, text
                nudges += 1
                if nudges > self.cfg.max_nudges:
                    raise AgentError("model did not call plan")
                msgs.append({"role": "user", "content": "Call plan now."})
                continue
            results, first_empty = [], False
            for call in calls:
                if call["name"] == "plan":
                    if used == 0:
                        out, err = "explore with tools before planning", True
                    else:
                        out, err = self._call(call)
                        if not err and not planned:
                            planned = True
                            first_empty = not self.plan["steps"]
                else:
                    out, err = self._call(call)
                    used += 1
                self._log(cycle, call, out, err)
                results.append({"type": "tool_result", "tool_use_id": call["id"], "content": out, "is_error": err})
            msgs.append({"role": "user", "content": results})
            if first_empty:
                return True, text
        return False, "step budget exhausted"

    def run(self, task):
        carry, last, repeats = "", None, 0
        for cycle in range(1, self.cfg.max_cycles + 1):
            print(f"\n=== cycle {cycle} ===", flush=True)
            msgs = [{"role": "user", "content": self._brief(task, cycle, carry)}]
            converged, summary = self._cycle(msgs, cycle)
            delta = self.plan["delta"]
            if converged:
                return f"DONE (cycle {cycle})\n{delta}"
            print(f"[{cycle}] {summary}", flush=True)
            repeats = repeats + 1 if delta == last else 0
            if repeats >= self.cfg.stall:
                return f"STALLED (cycle {cycle})\nunresolved delta: {delta}\nlast summary: {summary}"
            last = delta
            carry = f"delta: {delta}\nsteps: {'; '.join(self.plan['steps'])}\nresult: {clip(summary, 1500)}"
        return f"CYCLE LIMIT ({self.cfg.max_cycles})\nlast delta: {self.plan['delta']}"


def read_task():
    preset = _get("AGENT_TASK")
    if preset:
        return preset.strip()
    if not inColab and len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return input("task> ").strip()


def main():
    cfg = Config.load()
    workspace = Workspace(cfg)
    agent = ReAct(cfg, LLM(cfg), workspace.domain())
    print(f"mode={'colab' if inColab else 'ide'} model={cfg.model} root={cfg.root} tools={len(agent.tools)}")
    task = read_task()
    if not task:
        raise SystemExit("empty task")
    try:
        print("\n" + agent.run(task))
    except AgentError as exc:
        raise SystemExit(f"agent error: {exc}")
    except KeyboardInterrupt:
        raise SystemExit("interrupted")


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)

    def send(self, system, tools, messages):
        return self.script.pop(0)


def selftest():
    import tempfile

    def use(ident, name, **payload):
        return {"content": [{"type": "tool_use", "id": ident, "name": name, "input": payload}], "stop_reason": "tool_use"}

    def say(text):
        return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}

    def raises(fn):
        try:
            fn()
        except ToolError:
            return True
        return False

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config.load(llm=False, root=Path(tmp))
        ws = Workspace(cfg)
        mod = cfg.root / "pkg" / "mod.py"

        ws.create_file({"path": "pkg/mod.py", "content": "class A:\n    def f(self):\n        return 1\n\n    def g(self):\n        return 2\n"})
        assert "pkg/" in ws.map_repo({})
        assert "class A" in ws.analyse_file({"path": "pkg/mod.py"})
        assert "f" in ws.find_files({"pattern": "**/*.py"}) or "mod.py" in ws.find_files({"pattern": "*.py"})
        assert "pkg/mod.py:3:" in ws.grep({"pattern": "return 1"})
        assert "3|        return 1" in ws.read_file({"path": "pkg/mod.py", "start": 2, "end": 3})

        ws.patch_file({"path": "pkg/mod.py", "old": "def f(self):\n    return 1", "new": "def f(self):\n    value = 10\n    return value"})
        assert "    def f(self):\n        value = 10\n        return value\n" in mod.read_text()

        ws.patch_file({"path": "pkg/mod.py", "start": 1, "end": 0, "new": "import os"})
        assert mod.read_text().startswith("import os\nclass A:")

        snapshot = mod.read_text()
        assert raises(lambda: ws.patch_file({"path": "pkg/mod.py", "old": "return 2", "new": "return ("}))
        assert mod.read_text() == snapshot

        ws.create_file({"path": "nested.py", "content": "def f():\n    if a:\n        b()\n"})
        ws.patch_file({"path": "nested.py", "old": "if a:\n  b()", "new": "if a:\n  b()\n  c()"})
        assert (cfg.root / "nested.py").read_text() == "def f():\n    if a:\n        b()\n        c()\n"

        ws.create_file({"path": "tabs.txt", "content": "if x:\n\treturn 1\n"})
        ws.patch_file({"path": "tabs.txt", "old": "return 1", "new": "return 2"})
        assert (cfg.root / "tabs.txt").read_text() == "if x:\n\treturn 2\n"

        ws.create_file({"path": "dup.txt", "content": "x = 1\ny = 2\nx = 1\n"})
        assert raises(lambda: ws.patch_file({"path": "dup.txt", "old": "x = 1", "new": "x = 3"}))
        assert raises(lambda: ws.create_file({"path": "dup.txt", "content": "z"}))
        assert raises(lambda: ws.read_file({"path": "../outside"}))

        assert "+" in ws.state({})
        assert "no file changes" in ws.state({})
        assert "exit 0" in ws.run({"cmd": "echo ok"})
        assert "timeout" in ws.run({"cmd": "sleep 5", "timeout": 1})

        ws.delete_file({"path": "tabs.txt"})
        assert not (cfg.root / "tabs.txt").exists()

        # Groq wire conversion round trip
        system = [{"type": "text", "text": "sys"}]
        history = [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "t1", "name": "run", "input": {"cmd": "ls"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "exit 0", "is_error": False}]},
        ]
        wire = LLM._to_wire_messages(system, history)
        assert wire[0] == {"role": "system", "content": "sys"}
        assert wire[2]["tool_calls"][0]["function"]["name"] == "run"
        assert json.loads(wire[2]["tool_calls"][0]["function"]["arguments"]) == {"cmd": "ls"}
        assert wire[3] == {"role": "tool", "tool_call_id": "t1", "content": "exit 0"}
        parsed = LLM._from_wire({"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{"id": "x", "type": "function", "function": {"name": "grep", "arguments": "{\"pattern\": \"a\"}"}}]}}]})
        assert parsed["stop_reason"] == "tool_use" and parsed["content"][0]["input"] == {"pattern": "a"}
        assert LLM._from_wire({"choices": [{"finish_reason": "length", "message": {"content": "cut"}}]})["stop_reason"] == "max_tokens"
        schema = LLM._to_wire_tools(ws.domain().tools["grep"].schema() and [ws.domain().tools["grep"].schema()])
        assert schema[0]["function"]["parameters"]["type"] == "object"

        # SerpAPI response formatting and missing-key handling
        sample = {"organic_results": [{"title": "T", "link": "https://example.com", "snippet": "S"}], "answer_box": {"answer": "42"}}
        shown = Workspace._format_serp(sample, 5)
        assert "answer: 42" in shown and "1. T" in shown and "https://example.com" in shown
        assert raises(lambda: Workspace._format_serp({"error": "bad key"}, 5))
        if not cfg.serp_key:
            assert raises(lambda: ws.web_search({"query": "x"}))

        script = [
            use("a", "plan", delta="x", steps=[]),
            use("b", "map_repo"),
            use("c", "plan", delta="hello.py missing", steps=["create hello.py"]),
            use("d", "create_file", path="hello.py", content="print('hi')\n"),
            say("created hello.py"),
            use("e", "find_files", pattern="hello.py"),
            use("f", "plan", delta="none; hello.py exists", steps=[]),
        ]
        agent = ReAct(cfg, ScriptedLLM(script), ws.domain())
        result = agent.run("make hello.py")
        assert result.startswith("DONE (cycle 2)"), result
        assert (cfg.root / "hello.py").read_text() == "print('hi')\n"
        assert not agent.llm.script
        assert len(agent.tools) <= cfg.max_tools
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
