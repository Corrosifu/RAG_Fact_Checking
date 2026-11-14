"""Plain PyTorch fine-tuning loop for the NLI cross-encoder (CPU friendly, no Trainer)."""

from __future__ import annotations

import random
import shutil
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rag_fact_checking.training.pairs import Pair


def finetune(
    model_name: str,
    pairs: Sequence[Pair],
    out_dir: Path | str,
    evaluate: Callable[[Path], float],
    *,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_len: int = 192,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fine-tune ``model_name`` on ``pairs``; keep the epoch with the best ``evaluate`` score.

    ``evaluate(checkpoint_dir)`` returns a hold-out score (higher is better). The best
    checkpoint is copied to ``out_dir/best``. Class weights counter the neutral majority.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    out = Path(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    label2id = {str(k).lower(): v for k, v in model.config.label2id.items()}
    counts = Counter(p.label for p in pairs)
    weights = torch.tensor(
        [
            len(pairs) / (len(label2id) * max(counts.get(lab, 1), 1))
            for lab in sorted(label2id, key=label2id.get)
        ],
        dtype=torch.float,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(pairs) + batch_size - 1) // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    history: list[dict[str, float]] = []
    best_score, best_epoch = float("-inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(pairs)))
        random.shuffle(order)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = [pairs[i] for i in order[start : start + batch_size]]
            enc = tokenizer(
                [p.premise for p in batch],
                [p.claim for p in batch],
                truncation="only_first",
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            labels = torch.tensor([label2id[p.label] for p in batch])
            loss = loss_fn(model(**enc).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item() * len(batch)
        ckpt = out / f"epoch-{epoch}"
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        score = evaluate(ckpt)
        history.append(
            {"epoch": epoch, "train_loss": round(total / len(pairs), 4), "holdout": score}
        )
        log(f"epoch {epoch}: train loss {total / len(pairs):.4f}, hold-out score {score:.4f}")
        if score > best_score:
            best_score, best_epoch = score, epoch
    best = out / "best"
    if best.exists():
        shutil.rmtree(best)
    shutil.copytree(out / f"epoch-{best_epoch}", best)
    return {"history": history, "best_epoch": best_epoch, "best_holdout": round(best_score, 4)}
