import time
import csv
import re
from urllib import request
from html.parser import HTMLParser
from pathlib import Path

BASE_URL = "https://www.ptt.cc"

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

REQUIRED_TITLE_COUNT = 100000
MAX_PAGE_COUNT = 5050


class EarlyStop(Exception):
    pass


def get_html_content(URL: str, retries: int = 3) -> str | None:
    req = request.Request(
        URL,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
        },
    )
    for retry in range(retries):
        try:
            with request.urlopen(req) as f:
                content_charset = f.headers.get_content_charset()
                charset = content_charset if content_charset else "utf-8"
                html_content = f.read().decode(charset)
            return html_content
        except Exception as e:
            if retry >= retries - 1:
                print(f"\nError: {e}")
            else:
                time.sleep(0.5)
    return None


class NextPageLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._tag_attrs = None
        self.next_page_url = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a" and any(
            key == "class" and val is not None and "disabled" not in val
            for key, val in attrs
        ):
            self._tag_attrs = attrs

    def handle_data(self, data: str) -> None:
        if self._tag_attrs is not None and "上頁" in data:
            for key, val in self._tag_attrs:
                if key == "href" and val is not None:
                    self.next_page_url = val
                    raise EarlyStop

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._tag_attrs is not None:
            self._tag_attrs = None


class BoardTitleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._is_title = False
        self._is_title_link = False
        self._current_title_chunks = []
        self.title_list: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "div" and any(
            key == "class" and val is not None and "title" in val for key, val in attrs
        ):
            self._is_title = True
        elif self._is_title and tag == "a":
            self._is_title_link = True
            self._current_title_chunks = []

    def handle_data(self, data: str) -> None:
        if self._is_title and self._is_title_link:
            self._current_title_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._is_title_link and tag == "a":
            full_title = "".join(self._current_title_chunks).strip()
            if full_title:
                self.title_list.append(full_title)
            self._is_title_link = False
            self._current_title_chunks = []
        elif self._is_title and tag == "div":
            self._is_title = False


def data_crawler(board_name: str):
    next_page_url = f"{BASE_URL}/bbs/{board_name}/index.html"
    title_count = 0
    page_count = 0
    with open(f"{board_name}.csv", "w", newline="", encoding="utf-8") as f:
        while (
            title_count < REQUIRED_TITLE_COUNT
            and next_page_url is not None
            and page_count < MAX_PAGE_COUNT
        ):
            page_content = get_html_content(next_page_url)
            if page_content is None:
                break

            board_title_parser = BoardTitleParser()
            board_title_parser.feed(page_content)
            title_count += len(board_title_parser.title_list)
            writer = csv.writer(f)
            titles = board_title_parser.title_list
            writer.writerows([board_name, title.replace("\n", "")] for title in titles)
            print(
                f"\r{board_name} title count: {title_count}",
                end="",
                flush=True,
            )

            next_page_parser = NextPageLinkParser()
            try:
                next_page_parser.feed(page_content)
            except EarlyStop:
                pass
            if next_page_parser.next_page_url:
                next_page_url = f"{BASE_URL}{next_page_parser.next_page_url}"
            else:
                next_page_url = None

            page_count += 1
    print(f"\nThe board {board_name} finished with {title_count} titles")


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


def main():
    print("--- data crawler start ---")
    for board_name in BOARD_NAMES:
        data_crawler(board_name)
    print("--- data crawler done")
    print("--- data cleaner start ---")
    data_cleaner(
        [Path(f"{board_name}.csv") for board_name in BOARD_NAMES], Path("data.csv")
    )
    print("--- data cleaner done ---")


if __name__ == "__main__":
    main()
