import torch
from ckip_transformers.nlp import CkipPosTagger, CkipWordSegmenter

STOP_POS = {"Caa", "Cab", "Cba", "Cbb", "P", "T", "I", "Di", "Nh", "DE", "SHI"}


def is_stop_pos(pos: str) -> bool:
    return pos in STOP_POS or pos.endswith("CATEGORY") or pos == "WHITESPACE"


def tokenizer(
    input: list[str],
    device: torch.device = torch.device("cpu"),
) -> list[list[str]]:
    ws_driver = CkipWordSegmenter(model="bert-base", device=device)
    pos_driver = CkipPosTagger(model="bert-base", device=device)
    output: list[list[str]] = []
    ws_result = ws_driver(input, batch_size=256)
    pos_result = pos_driver(ws_result, batch_size=256)
    for ws_list, pos_list in zip(ws_result, pos_result):
        ws_output: list[str] = []
        for ws, pos in zip(ws_list, pos_list):
            if is_stop_pos(pos) or not ws.strip():
                continue
            ws_output.append(ws.strip())
        if not ws_output:
            continue
        output.append(ws_output)
    return output
