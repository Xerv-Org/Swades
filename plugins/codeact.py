import io
import sys
import traceback

VERSION = "1.0"

PROMPT = "Use the python tool to execute Python code in a persistent session. Use print() to see output. Write code to compute, filter, or process multi-step tasks in one turn."


def setup(api):
    scope = {"__builtins__": __builtins__}
    limit = api.num("CODEACT_CHARS", 8000)

    def execute(a):
        code = str(a["code"])
        buf_out, buf_err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = buf_out, buf_err
            exec(code, scope)
        except Exception:
            traceback.print_exc(file=buf_err)
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        out, err = buf_out.getvalue(), buf_err.getvalue()
        combined = (out + ("\nSTDERR:\n" + err if err else "")).strip()
        return api.clip(combined, limit) if combined else "(executed with no output)"

    api.tool("python", "Execute Python code in a persistent session. Output from print() is returned.", {"code": api.need()}, execute)
    api.prompt(PROMPT)
