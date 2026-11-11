"""Test doubles shared across test modules."""


class StubVerifier:
    """Deterministic NLI stand-in.

    Entails premises that mention ``yes_word`` (when the claim does too) and contradicts
    any claim containing ``no_word``.
    """

    def __init__(self, yes_word="quantization", no_word="never"):
        self.yes_word, self.no_word = yes_word, no_word
        self.calls = []
        self.premise_texts = []

    def entailment_probs(self, claim, premises):
        self.calls.append((claim, len(premises)))
        self.premise_texts.extend(p.text for p in premises)
        out = []
        for p in premises:
            if self.no_word in claim.lower():
                out.append({"entailment": 0.02, "contradiction": 0.95, "neutral": 0.03})
            elif self.yes_word in p.text.lower() and self.yes_word in claim.lower():
                out.append({"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03})
            else:
                out.append({"entailment": 0.1, "contradiction": 0.05, "neutral": 0.85})
        return out
