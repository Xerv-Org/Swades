import os

VERSION = "1.0"

PROMPT = (
    "Memory: your only memory across runs is the notes shown in STATE. Before finishing, call notes_write with the complete, compact, current notes: facts, decisions, open items, next actions. "
    "Drop stale entries and stay under the size limit."
)


def setup(api):
    path = api.data / "NOTES.md"
    limit = api.num("NOTES_CHARS", 12000)

    def read():
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write(a):
        text = str(a["text"]).strip()
        if len(text) > limit:
            raise api.ToolError(f"{len(text)} chars exceeds limit {limit}; rewrite shorter")
        tmp = path.with_name("NOTES.md.tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return f"saved {len(text)} chars"

    api.tool("notes_write", "Replace the persistent notes with the full new text.", {"text": api.need()}, write)
    api.prompt(PROMPT)
    api.state(lambda: "NOTES\n" + (read().strip() or "empty; call notes_write to start memory"))
