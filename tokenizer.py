import csv
from pathlib import Path

import torch
from ckip_transformers.nlp import CkipPosTagger, CkipWordSegmenter

DEFAULT_STOP_POS = {"Caa", "Cab", "Cba", "Cbb", "P", "T", "I", "Di", "Nh", "DE", "SHI"}


def is_stop_pos(
    pos: str,
    stop_pos: set[str] | list[str] = DEFAULT_STOP_POS,
    allowed_pos: set[str] | list[str] | None = None,
) -> bool:
    if pos.endswith("CATEGORY") or pos == "WHITESPACE":
        return True
    if allowed_pos is not None:
        return pos not in allowed_pos
    return pos in stop_pos


default_device = torch.device("cpu")


def tokenizer(
    input: Path,
    output: Path,
    stop_pos: set[str] | list[str] = DEFAULT_STOP_POS,
    allowed_pos: set[str] | list[str] | None = None,
    min_tokens: int = 1,
    chunk_size: int = 4096,
    batch_size: int = 256,
    device: torch.device = default_device,
):
    ws_driver = CkipWordSegmenter(model="bert-base", device=device)
    pos_driver = CkipPosTagger(model="bert-base", device=device)
    stop_pos_set = set(stop_pos)
    allowed_pos_set = set(allowed_pos) if allowed_pos is not None else None

    def tokenize(docs: list[str]):
        ws_result = ws_driver(docs, batch_size=batch_size)
        pos_result = pos_driver(ws_result, batch_size=batch_size)
        result: list[list[str]] = []
        for ws_list, pos_list in zip(ws_result, pos_result):
            tokens: list[str] = []
            for w, p in zip(ws_list, pos_list):
                if is_stop_pos(p, stop_pos=stop_pos_set, allowed_pos=allowed_pos_set) or not w.strip():
                    continue
                tokens.append(w.strip())
            result.append(tokens)
        return result

    with (
        open(input, "r", encoding="utf-8") as f_in,
        open(output, "w", newline="", encoding="utf-8") as f_out,
    ):
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        def write(labels: list[str], tokens_list: list[list[str]]):
            for label, tokens in zip(labels, tokens_list):
                if len(tokens) >= min_tokens:
                    writer.writerow([label, *tokens])

        batch_boards = []
        batch_titles = []
        for row in reader:
            if not row or len(row) < 2:
                continue
            board, title = row[0], row[1]
            batch_boards.append(board)
            batch_titles.append(title)

            if len(batch_titles) >= chunk_size:
                tokens_list = tokenize(batch_titles)
                write(batch_boards, tokens_list)
                batch_boards.clear()
                batch_titles.clear()

        if batch_titles:
            tokens_list = tokenize(batch_titles)
            write(batch_boards, tokens_list)
