"""Long-term memory: short facts the owner asked the robot to remember ("our pit is A3").

Stored on the robot in ~/.reachy_gemini_dj/memory.json. Facts are given to the local brain (hello message) and to
Gemini Live (system instruction). Memory is personal: fact text is never logged, only counts.

Tool results carry a complete, short, natural "say" that the local brain speaks directly (no LLM round).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

from .config import HOME

log = logging.getLogger(__name__)

MEMORY_FILE = HOME / "memory.json"
MAX_FACTS = 60
MAX_CHARS = 200
NEAR_DUPLICATE = 0.8            # word-set overlap (Jaccard) at or above which a new fact counts as already known
MAX_SPOKEN = 8                  # list_memories reads out at most this many

_STOP = {"the", "a", "an", "our", "my", "your", "their", "his", "her", "its", "is", "are", "was", "were", "to", "of",
         "that", "this", "it", "and", "i", "me", "you", "we", "in", "on", "at", "for", "about", "s", "now",
         "currently", "also", "please", "remember", "forget"}
_COPULA = {"is", "are", "was", "were"}
_SUBJECT_STOP = {"the", "a", "an", "our", "my", "your", "their", "his", "her", "its", "s"}
_ALL_WORDS = {"all", "everything"}
_ALL_FILLER = {"all", "everything", "of", "it", "them", "my", "the", "your", "you", "memories", "memory", "things",
               "facts", "that", "i", "told", "asked", "to", "remember", "forget", "stuff", "these", "those", "we"}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:MAX_CHARS].strip()


def _words(text: str) -> list[str]:
    """Lowercase, punctuation-free words (apostrophes dropped: "team's" -> "teams")."""
    return re.sub(r"[^a-z0-9 ]", " ", re.sub(r"['’]", "", str(text).lower())).split()


def norm(text: str) -> str:
    return " ".join(_words(text))


def subject_key(text: str) -> str:
    """Words before the first "is/are/was/were" (within the first 6 words), minus articles and possessives.

    "our pit is A3" / "The team's pit is now B7" -> "pit"; "my name is Sam" -> "name". Empty when the fact is not
    of the "<thing> is <value>" shape, so it can never replace anything. "team's"/"robot's" style possessives are
    dropped so the same thing said two ways matches; other qualifiers ("dog name") are kept so they stay separate.
    """
    t = re.sub(r"\b(team|robot|owner|user|reachy)['’]s\b", " ", str(text).lower())
    w = _words(re.sub(r"['’]", " ", t))
    for i, word in enumerate(w[:6]):
        if word in _COPULA and i > 0:
            key = [x for x in w[:i] if x not in _SUBJECT_STOP]
            return " ".join(key)
    return ""


def _content(text: str) -> set[str]:
    out = set()
    for w in _words(text):
        if w in _STOP:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def _lower_first(s: str) -> str:
    first = s.split(" ", 1)[0]
    if not s or (len(s) > 1 and s[1].isupper()) or first == "I" or first.startswith(("I'", "I’")):
        return s                                # acronyms and "I ..." keep their capital
    return s[0].lower() + s[1:]


def _spoken(fact: str) -> str:
    return fact.rstrip(". ")


class Memory:
    """Thread-safe list of {id, text, created} facts, persisted atomically."""

    def __init__(self, path: Path | str = MEMORY_FILE):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._facts: list[dict] = []
        self._load()

    # ----- persistence -----
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            items = data.get("facts") if isinstance(data, dict) else data
            for it in items or []:
                text = clean_text(it.get("text", "")) if isinstance(it, dict) else ""
                if text:
                    self._facts.append({"id": str(it.get("id") or uuid.uuid4().hex[:8]), "text": text,
                                        "created": str(it.get("created", ""))})
            self._facts = self._facts[:MAX_FACTS]
        except FileNotFoundError:
            pass
        except Exception as e:
            log.warning("Could not read memory file (%s); starting with no memories", type(e).__name__)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"facts": self._facts}, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)              # atomic: a crash never leaves a half-written file

    # ----- reading -----
    def texts(self) -> list[str]:
        with self._lock:
            return [f["text"] for f in self._facts]

    def listing(self) -> list[dict]:
        with self._lock:
            return [dict(f) for f in self._facts]

    def __len__(self) -> int:
        with self._lock:
            return len(self._facts)

    # ----- writing -----
    def add(self, text: str) -> dict:
        """Store a fact. Returns {"status": added|updated|duplicate|empty|full, "fact": {...}|None, "old": text|None}."""
        text = clean_text(text)
        if not norm(text):
            return {"status": "empty", "fact": None, "old": None}
        with self._lock:
            n, key = norm(text), subject_key(text)
            nset = set(n.split())
            for f in self._facts:
                fn = norm(f["text"])
                fset = set(fn.split())
                if fn == n or (fset and nset and len(fset & nset) / len(fset | nset) >= NEAR_DUPLICATE):
                    return {"status": "duplicate", "fact": dict(f), "old": None}
            new = {"id": uuid.uuid4().hex[:8], "text": text, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
            if key:
                for i, f in enumerate(self._facts):
                    if subject_key(f["text"]) == key:
                        old = f["text"]
                        self._facts[i] = new                 # same thing, new value: replace in place
                        self._write()
                        log.info("memory updated (%d facts)", len(self._facts))
                        return {"status": "updated", "fact": dict(new), "old": old}
            if len(self._facts) >= MAX_FACTS:
                return {"status": "full", "fact": None, "old": None}
            self._facts.append(new)
            self._write()
            log.info("memory added (%d facts)", len(self._facts))
            return {"status": "added", "fact": dict(new), "old": None}

    def delete_id(self, fact_id: str) -> bool:
        with self._lock:
            keep = [f for f in self._facts if f["id"] != str(fact_id)]
            if len(keep) == len(self._facts):
                return False
            self._facts = keep
            self._write()
            log.info("memory deleted (%d facts left)", len(self._facts))
            return True

    def clear(self) -> int:
        with self._lock:
            n = len(self._facts)
            self._facts = []
            self._write()
            log.info("memory cleared (%d facts removed)", n)
            return n

    def match(self, query: str) -> list[dict]:
        """Facts best matching the query by word overlap: all those tied for the top score, if it is good enough."""
        q = norm(query)
        qc = _content(query)
        with self._lock:
            exact = [f for f in self._facts if norm(f["text"]) == q]
            if exact:
                return [dict(exact[0])]
            if not qc:
                return []
            scored = []
            for f in self._facts:
                overlap = len(qc & _content(f["text"]))
                if overlap:
                    scored.append((overlap / len(qc), overlap, f))
            if not scored:
                return []
            best = max(s for s, _, _ in scored)
            if best < 0.5:
                return []
            top = [(o, f) for s, o, f in scored if s == best]
            return [dict(f) for _, f in top]

    # ----- tool handlers (return dicts with a spoken "say") -----
    def tool_remember(self, fact: str) -> dict:
        r = self.add(fact)
        st, f = r["status"], r["fact"]
        if st == "empty":
            return {"error": "Nothing to remember: give me the fact."}
        if st == "full":
            return {"error": f"Memory is full ({MAX_FACTS} things). Ask me to forget something first.",
                    "say": "My memory is full. Tell me something to forget first."}
        what = _lower_first(_spoken(f["text"]))
        if st == "duplicate":
            return {"result": "already known", "say": "I already know that."}
        if st == "updated":
            return {"result": "updated", "say": f"Okay, I updated that: {what}."}
        return {"result": "remembered", "say": f"Got it, I'll remember that {what}."}

    def tool_forget(self, fact: str) -> dict:
        q = clean_text(fact)
        if not norm(q):
            return {"error": "Say what to forget."}
        w = set(_words(q))
        if (w & _ALL_WORDS) and w <= _ALL_FILLER:
            n = self.clear()
            return {"result": f"forgot {n}", "say": "Okay, I forgot everything." if n else "I wasn't remembering anything."}
        hits = self.match(q)
        if not hits:
            return {"result": "no match", "say": "I don't remember anything like that."}
        if len(hits) > 1:
            opts = [_spoken(h["text"]) for h in hits[:3]]
            joined = ", or ".join([", ".join(opts[:-1]), opts[-1]]) if len(opts) > 2 else " or ".join(opts)
            return {"result": "ambiguous", "say": f"Which one: {joined}?"}
        self.delete_id(hits[0]["id"])
        return {"result": "forgot", "say": "Okay, I forgot that."}

    def tool_list(self) -> dict:
        facts = self.texts()
        if not facts:
            return {"result": "empty", "memories": [], "say": "I'm not remembering anything yet."}
        shown = [_spoken(t) for t in facts[:MAX_SPOKEN]]
        more = len(facts) - len(shown)
        head = "I remember one thing" if len(facts) == 1 else f"I remember {len(facts)} things"
        say = f"{head}: " + "; ".join(shown) + (f"; and {more} more." if more else ".")
        return {"result": "ok", "count": len(facts), "memories": facts, "say": say}


def live_instruction_line(facts: list[str]) -> str:
    """One plain line for the Gemini Live system instruction ('' when there is nothing to remember)."""
    facts = [_spoken(clean_text(t)) for t in facts if clean_text(t)]
    return f"Things you were asked to remember: {'; '.join(facts)}." if facts else ""
