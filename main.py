from pathlib import Path
import torch
from ckip_transformers.nlp import CkipPosTagger, CkipWordSegmenter
from crawler import crawler as data_crawler
from cleaner import data_cleaner

BOARD_NAMES = [
    "Baseball",
    "Boy-Girl",
    "C_Chat",
    "HatePolitics",
    "Lifeismoney",
    "Military",
    "PC_Shopping",
    "Stock",
    "Tech_Job",
]

REQUIRED_TITLE_COUNT = 100


def get_device():
    device = (
        acc
        if (acc := torch.accelerator.current_accelerator(check_available=True))
        is not None
        else torch.device("cpu")
    )
    print(f"Using {device.type} device")
    return device


def tokenizer():
    device = get_device()
    ws_driver = CkipWordSegmenter(model="bert-base", device=device)
    pos_driver = CkipPosTagger(model="bert-base", device=device)
    text_list = [
        "前端工程師常用 Vue 與 TypeScript 開發網頁。",
        "今天淡水天氣很好，適合出門散步。",
    ]
    ws_result = ws_driver(text_list)
    pos_result = pos_driver(ws_result)
    for ws_list, pos_list in zip(ws_result, pos_result):
        for ws, pos in zip(ws_list, pos_list):
            print(ws, pos)
        print("---")
    pass


def main():
    print("--- data crawler start ---")
    for board_name in BOARD_NAMES:
        data_crawler(board_name, REQUIRED_TITLE_COUNT)
    print("--- data crawler done")
    print("--- data cleaner start ---")
    data_cleaner(
        [Path(f"{board_name}.csv") for board_name in BOARD_NAMES], Path("raw_data.csv")
    )
    print("--- data cleaner done ---")
    # tokenizer()


if __name__ == "__main__":
    main()
