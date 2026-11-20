"""Token-budgeted views with exact source offsets and complete coverage."""

from __future__ import annotations


def text_views(
    text: str, tokenizer, max_tokens: int = 2048, overlap_tokens: int = 128
) -> list[dict]:
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap must be smaller than the text token budget")
    if not text:
        return []
    tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = tokens["offset_mapping"]
    if not offsets:
        return [
            {"char_start": 0, "char_end": len(text), "token_count": 0, "text": text}
        ]
    views = []
    start = 0
    while start < len(offsets):
        end = min(start + max_tokens, len(offsets))
        char_start = 0 if start == 0 else offsets[start][0]
        char_end = len(text) if end == len(offsets) else offsets[end - 1][1]
        content = text[char_start:char_end]
        actual_count = len(tokenizer.encode(content, add_special_tokens=False))
        while actual_count > max_tokens and end > start + 1:
            end -= 1
            char_end = offsets[end - 1][1]
            content = text[char_start:char_end]
            actual_count = len(tokenizer.encode(content, add_special_tokens=False))
        if actual_count > max_tokens:
            raise ValueError("one source character exceeds the configured token budget")
        views.append(
            {
                "char_start": char_start,
                "char_end": char_end,
                "token_count": actual_count,
                "text": content,
            }
        )
        if char_end == len(text):
            break
        start = max(start + 1, end - overlap_tokens)
    covered = 0
    for view in views:
        if view["char_start"] > covered:
            raise ValueError("tokenizer offsets leave an uncovered source interval")
        covered = max(covered, view["char_end"])
    if covered != len(text):
        raise ValueError("text view plan has incomplete coverage")
    return views
