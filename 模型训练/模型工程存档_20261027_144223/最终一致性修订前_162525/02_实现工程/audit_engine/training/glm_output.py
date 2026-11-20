"""Extract GLM's native final-answer channel without repairing its content."""

THINK_START = 151345
THINK_END = 151346
ANSWER_START = 151347
ANSWER_END = 151348


def final_answer_ids(ids):
    ids = list(ids)
    if ANSWER_START in ids:
        start = ids.index(ANSWER_START) + 1
    elif THINK_END in ids:
        start = ids.index(THINK_END) + 1
    elif THINK_START in ids:
        return []
    else:
        start = 0
    end = ids.index(ANSWER_END, start) if ANSWER_END in ids[start:] else len(ids)
    return ids[start:end]
