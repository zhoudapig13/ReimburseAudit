"""Bind unchanged model quotations to literal segments in the supplied view."""


def literal_source_refs(evidence_id, segments, quote, preferred_segment=None):
    normalize = lambda value: "".join(value.split())
    remaining = normalize(quote)
    if not remaining:
        raise ValueError("source quotation is empty")
    normalized = [(segment, normalize(segment["text"])) for segment in segments]

    def citation(segment):
        return {"evidence_id": evidence_id, "segment_id": segment["segment_id"],
                "quote": segment["text"], "kind": "text",
                "note": "Literal source binding; source text and model-read value unchanged."}

    direct = [segment for segment, text in normalized if remaining in text]
    if direct:
        preferred = [segment for segment in direct if segment["segment_id"] == preferred_segment]
        if len(preferred) == 1:
            return [citation(preferred[0])]
        if len(direct) != 1:
            raise ValueError("source quotation occurs in multiple segments without an unambiguous segment id")
        return [citation(direct[0])]
    references = {}
    previous_index = None
    for _ in range(4):
        candidates = []
        for index, (segment, text) in enumerate(normalized):
            if previous_index is not None and index != previous_index + 1:
                continue
            low, high = 0, len(remaining)
            while low < high:
                middle = (low + high + 1) // 2
                if remaining[:middle] in text:
                    low = middle
                else:
                    high = middle - 1
            candidates.append((low, segment))
        maximum = max((length for length, _ in candidates), default=0)
        best = [segment for length, segment in candidates if length == maximum]
        if maximum < 8 or len(best) != 1:
            raise ValueError("quotation cannot be fully and unambiguously bound to literal source spans")
        segment = best[0]
        previous_index = next(index for index, (item, _) in enumerate(normalized)
                              if item["segment_id"] == segment["segment_id"])
        references[segment["segment_id"]] = citation(segment)
        remaining = remaining[maximum:]
        if not remaining:
            return list(references.values())
    raise ValueError("quotation spans too many original segments")
