"""Build prototype/index.html from template.html + results/results.json (run `make results` first).

    python prototype/build_prototype.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(out: Path | None = None) -> Path:
    res = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    tpl = (ROOT / "prototype" / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("__DATA__", json.dumps(res, ensure_ascii=False, separators=(",", ":")))
    out = out or ROOT / "prototype" / "index.html"
    out.write_text(html, encoding="utf-8")
    print("built", out, len(html), "bytes")
    return out


if __name__ == "__main__":
    main()
