"""ahaloop command line interface."""

from __future__ import annotations

import typer

from ahaloop.transcript import TranscriptError, fetch_transcript, source_ref

app = typer.Typer(help="Turn any video into a learning loop.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Keep subcommands addressable by name.

    Typer promotes a lone command to the root of the CLI, which would make this
    `ahaloop <url>` today and silently rename it to `ahaloop ingest <url>` the
    moment a second command lands. Declaring a callback pins the multi-command
    shape now, so the documented invocation never changes under users.
    """


@app.command(help="Turn a YouTube video into a learning pack.")
def ingest(url: str) -> None:
    """Turn a YouTube video into a learning pack.

    Fetches the transcript today. Still planned for M1:
        1. Segment the transcript into concepts.
        2. Generate a learning pack (concept notes + retrieval quiz) via
           `ahaloop.llm.get_client`, citing the timestamps it came from.
        3. Emit `gap_opened` / `quiz_result` events through `ahaloop.events`.

    Args:
        url: A YouTube video URL.
    """
    try:
        transcript = fetch_transcript(url)
    except TranscriptError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    kind = "auto-generated" if transcript.is_generated else "human-written"
    typer.echo(
        f"{transcript.video_id}: {len(transcript.segments)} segments "
        f"({transcript.language_code}, {kind})"
    )
    for segment in transcript.segments[:3]:
        typer.echo(f"  {source_ref(transcript.video_id, segment)}  {segment.text}")


if __name__ == "__main__":
    app()
