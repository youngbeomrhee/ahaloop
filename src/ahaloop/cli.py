"""ahaloop command line interface."""

from __future__ import annotations

import typer

from ahaloop.llm import LLMConfigurationError, get_client
from ahaloop.packs import extract_pack
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
def ingest(url: str, pack: bool = False) -> None:
    """Turn a YouTube video into a learning pack.

    Fetches the transcript, and with `--pack` extracts cited concepts from it.
    Still planned for M1:
        1. Turn the concepts into a retrieval quiz.
        2. Emit `gap_opened` / `quiz_result` events through `ahaloop.events`.

    Args:
        url: A YouTube video URL.
        pack: Also run extraction, which requires a configured LLM provider.
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

    if not pack:
        return

    try:
        report = extract_pack(get_client(), transcript)
    except LLMConfigurationError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("")
    for concept in report.concepts:
        typer.echo(f"  {concept.concept}")
        typer.echo(f"    {concept.summary}")
        typer.echo(f"    {concept.source_ref}")
    typer.secho(
        f"\n{len(report.concepts)} concepts kept, {len(report.discarded)} discarded "
        f"(ungrounded rate {report.ungrounded_rate:.1%})",
        fg=typer.colors.YELLOW if report.discarded else typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
