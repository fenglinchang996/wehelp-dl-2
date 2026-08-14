import csv
import re
from pathlib import Path


def clean_title(title: str) -> str | None:
    cleaned_title = title.strip().lower()
    if cleaned_title.startswith("re:") or cleaned_title.startswith("fw:"):
        return None
    cleaned_title = re.sub(r"^(?:\[.*?\]\s*)+", "", cleaned_title)
    if not cleaned_title:
        return None
    return cleaned_title


def data_cleaner(input_files: list[Path], output_file: Path):
    with open(output_file, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)

        for input_file in input_files:
            if input_file.is_file():
                with open(input_file, "r", encoding="utf-8") as f_in:
                    reader = csv.reader(f_in)

                    for row in reader:
                        if not row or len(row) < 2:
                            continue
                        board, raw_title = row[0], row[1]
                        cleaned_title = clean_title(raw_title)
                        if cleaned_title:
                            writer.writerow([board, cleaned_title])
