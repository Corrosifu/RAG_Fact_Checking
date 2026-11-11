"""Answer generators.

* :class:`ExtractiveGenerator` – no model: picks the evidence sentences that best
  match the question and cites them. Instant and runs anywhere.
* :class:`HFGenerator` – a small instruction-tuned causal LM via ``transformers``
  (``ml`` extra). CPU by default; uses the GPU automatically if one is present.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Protocol

from rag_fact_checking.bm25 import tokenize
from rag_fact_checking.config import Settings
from rag_fact_checking.embeddings import STOPWORDS
from rag_fact_checking.schemas import Hit

NO_EVIDENCE = "I could not find relevant evidence in the knowledge base."

SYSTEM_PROMPT = (
    "You are a careful scientific assistant. Answer the question using ONLY the numbered "
    "context passages. Cite passages like [1]. If the context does not contain the answer, "
    "say so. Explain in your own words, clearly and concisely."
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _stem(token: str) -> str:
    """Crude plural/3rd-person stripping so 'reduces' matches 'reduce'."""
    return (
        token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token
    )


_MARKUP_RE = re.compile(r"[#*_`>|]+|\[\d+(?:[,\s-]+\d+)*\]")


def _clean_sentence(sentence: str) -> str:
    """Drop markdown markup and bare citation markers like [11] from extracted PDF text."""
    return " ".join(_MARKUP_RE.sub(" ", sentence).split())


def _strip_title(text: str, title: str) -> str:
    """Remove the (whitespace-normalised) document title, which PDFs repeat as page headers."""
    title = " ".join(title.split())
    return re.sub(re.escape(title), " ", text, flags=re.IGNORECASE) if title else text


def _usable(sentence: str) -> bool:
    """Keep prose-like sentences: not shouting caps, not a fragment."""
    words = sentence.split()
    letters = [c for c in sentence if c.isalpha()]
    if len(words) < 5 or len(sentence) > 500 or not letters:
        return False
    return sum(c.isupper() for c in letters) / len(letters) <= 0.5


def _terms(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text) if t not in STOPWORDS}


class Generator(Protocol):
    def generate(self, query: str, hits: Sequence[Hit], max_new_tokens: int = 256) -> str: ...


def build_context(hits: Sequence[Hit], max_chars: int = 3000) -> str:
    """Numbered passages, truncated to ``max_chars`` characters in total."""
    parts: list[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        text = " ".join(hit.chunk.text.split())
        remaining = max_chars - used
        if remaining <= 0:
            break
        piece = f"[{i}] {text[:remaining]}"
        parts.append(piece)
        used += len(piece) + 1
    return "\n".join(parts)


def build_messages(query: str, hits: Sequence[Hit], max_chars: int = 3000) -> list[dict[str, str]]:
    user = f"Context:\n{build_context(hits, max_chars)}\n\nQuestion: {query}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class ExtractiveGenerator:
    def __init__(self, max_sentences: int = 3, min_relative_score: float = 0.6) -> None:
        self.max_sentences = max_sentences
        self.min_relative_score = min_relative_score

    def generate(self, query: str, hits: Sequence[Hit], max_new_tokens: int = 256) -> str:
        q_terms = _terms(query)
        candidates: list[tuple[int, int, str, set[str]]] = []
        seen: set[str] = set()
        for h_idx, hit in enumerate(hits):
            for s_idx, sent in enumerate(
                _SENTENCE_RE.split(_strip_title(_clean_sentence(hit.chunk.text), hit.chunk.title))
            ):
                terms = _terms(sent)
                key = sent.lower()
                # overlapping chunks repeat sentences: keep the best-ranked occurrence only
                if key in seen or not _usable(sent) or not q_terms & terms:
                    continue
                seen.add(key)
                candidates.append((h_idx, s_idx, sent, terms))
        if not candidates:
            return NO_EVIDENCE

        # Rare query terms (e.g. "quantization") matter more than common ones ("model").
        n = len(candidates)
        idf = {t: math.log(1 + n / max(1, sum(t in c[3] for c in candidates))) for t in q_terms}
        scored = [
            (
                sum(idf[t] for t in q_terms & terms) / math.sqrt(len(terms)),
                -h,
                -s,
                f"{sent} [{h + 1}]",
            )
            for h, s, sent, terms in candidates
        ]
        scored.sort(reverse=True)  # ties favour better-ranked hits, then earlier sentences
        best = scored[0][0]
        picked = [
            t[3] for t in scored[: self.max_sentences] if t[0] >= self.min_relative_score * best
        ]
        return " ".join(picked)


class HFGenerator:
    def __init__(
        self,
        model_name: str,
        max_context_chars: int = 3000,
        token: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "transformers/torch are not installed. Run: pip install 'rag-fact-checking[ml]'"
            ) from exc
        self._torch = torch
        self.max_context_chars = max_context_chars
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            token=token,
        ).to(self.device)
        self.model.eval()

    def generate(  # pragma: no cover - needs model
        self, query: str, hits: Sequence[Hit], max_new_tokens: int = 256
    ) -> str:
        if not hits:
            return NO_EVIDENCE
        messages = build_messages(query, hits, self.max_context_chars)
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_generator(settings: Settings) -> Generator:
    if settings.generator == "extractive":
        return ExtractiveGenerator()
    if settings.generator == "hf":
        return HFGenerator(settings.generator_model, settings.max_context_chars, settings.hf_token)
    raise ValueError(f"Unknown generator {settings.generator!r} (use 'extractive' or 'hf')")
