"""ahaloop command line interface."""

from __future__ import annotations

import typer

app = typer.Typer(help="Turn any video into a learning loop.", no_args_is_help=True)


@app.command(help="Turn a YouTube video into a learning pack.")
def ingest(url: str) -> None:
    """Turn a YouTube video into a learning pack.

    Planned for M1:
        1. Fetch the transcript for `url` and keep timestamps as citation anchors.
        2. Segment the transcript into concepts.
        3. Generate a learning pack (concept notes + retrieval quiz) via
           `ahaloop.llm.get_client`, citing the timestamps it came from.
        4. Emit `gap_opened` / `quiz_result` events through `ahaloop.events`.

    Args:
        url: A YouTube video URL.
    """
    raise NotImplementedError("ingest lands in M1")


if __name__ == "__main__":
    app()
