"""
Transcript reformatting pipeline
--------------------------------

This script converts all Word transcripts (`.docx`) in the
`data/input/word_docs` directory into Excel workbooks in
`data/input/excel_format`, with the following columns:

- timestamp
- speaker
- utterance

The expected transcript structure (per block) is:

    00:00:02 Speaker 1
    First part of the utterance...
    Possible continuation lines...

Each timestamp/speaker line is followed by one or more lines of text
belonging to that utterance, until the next timestamp line or the end
of the document.

Usage from the command line:

    # From the project root:
    python transcript_reformat.py

Dependencies (install once):

    pip install python-docx pandas openpyxl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from docx import Document


TIMESTAMP_SPEAKER_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<timestamp>\d{2}:\d{2}:\d{2})   # 00:00:02
    \s+
    (?P<speaker>.+?)                   # Speaker 1 / Speaker 2 / Name
    \s*$
    """,
    re.VERBOSE,
)


def read_docx_lines(path: Path) -> List[str]:
    """
    Read a .docx file and return a list of paragraph texts (one per line).
    Empty paragraphs are kept as empty strings to preserve structure.
    """
    document = Document(str(path))
    lines = [p.text.strip() for p in document.paragraphs]
    return lines


def parse_transcript_lines(lines: Iterable[str]) -> pd.DataFrame:
    """
    Parse transcript lines into a DataFrame with columns:
    ['timestamp', 'speaker', 'utterance'].

    The function:
    - Detects lines matching TIMESTAMP_SPEAKER_PATTERN as the start of a block.
    - Concatenates following non-empty lines as the utterance until the next block.
    - Ignores any content before the first timestamp line.
    """
    records: List[Tuple[str, str, str]] = []

    current_timestamp: str | None = None
    current_speaker: str | None = None
    current_utterance_lines: List[str] = []

    def flush_current():
        """Append the current buffer as a record (if any)."""
        nonlocal current_timestamp, current_speaker, current_utterance_lines
        if current_timestamp is None or current_speaker is None:
            return
        utterance = " ".join(
            line.strip()
            for line in current_utterance_lines
            if line.strip()
        )
        records.append((current_timestamp, current_speaker, utterance))
        current_timestamp = None
        current_speaker = None
        current_utterance_lines = []

    for raw_line in lines:
        line = raw_line.strip()

        # Check if this line starts a new block (timestamp + speaker)
        match = TIMESTAMP_SPEAKER_PATTERN.match(line)
        if match:
            # Finish any previous block
            flush_current()

            current_timestamp = match.group("timestamp")
            current_speaker = match.group("speaker")
            current_utterance_lines = []
            continue

        # Otherwise, treat the line as part of the current utterance
        if current_timestamp is not None and current_speaker is not None:
            # We include even empty lines; they are filtered in flush_current
            current_utterance_lines.append(raw_line)
        else:
            # Content before the first timestamp is ignored (e.g., headers)
            continue

    # Flush final record
    flush_current()

    df = pd.DataFrame(records, columns=["timestamp", "speaker", "utterance"])
    return df


def process_transcript_docx(
    input_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    High-level pipeline:
    1. Read .docx into lines.
    2. Parse into timestamp/speaker/utterance table.
    3. Write an Excel file to `output_dir` with the same name as the
       original `.docx`, but with an `.xlsx` extension.

    Returns the resulting DataFrame.
    """
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".docx":
        raise ValueError(f"Input file must be a .docx file, got: {input_path.suffix}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = read_docx_lines(input_path)
    df = parse_transcript_lines(lines)

    # Output path: same file name, but in output_dir and with .xlsx extension
    xlsx_path = output_dir / (input_path.stem + ".xlsx")

    # Write output (overwrite if it already exists)
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    return df


def main() -> None:
    """
    Loop over all .docx files in `data/input/word_docs` and write the
    corresponding Excel files to `data/input/excel_format`, using the
    same base file names.
    """
    base_dir = Path(__file__).resolve().parent
    word_dir = base_dir / "data" / "input" / "word_docs"
    excel_dir = base_dir / "data" / "input" / "excel_format"

    if not word_dir.exists():
        raise FileNotFoundError(f"Word transcripts directory not found: {word_dir}")

    docx_files = sorted(word_dir.glob("*.docx"))
    if not docx_files:
        print(f"No .docx files found in {word_dir}")
        return

    print(f"Found {len(docx_files)} .docx file(s) in {word_dir}")
    print(f"Writing Excel files to {excel_dir}")

    for path in docx_files:
        print(f"- Processing {path.name} ...", end="", flush=True)
        df = process_transcript_docx(path, output_dir=excel_dir)
        print(f" done ({len(df)} rows)")


if __name__ == "__main__":
    main()


