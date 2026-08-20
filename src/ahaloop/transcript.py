"""YouTube transcript ingestion — the entry point of every learning pack.

The transcript is kept as timed segments rather than one flat string, because
every downstream claim in a learning pack must be traceable back to the moment
it was said. Flattening first and re-deriving timings later is not possible:
the mapping is lost the moment the segments are joined.

Auto-generated captions do not respect sentence boundaries — a segment routinely
ends mid-clause. Nothing here repairs that. Re-assembly is a decision for the
segmentation step, which needs the raw boundaries intact to make it.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi


class TranscriptError(RuntimeError):
    """Raised when a URL yields no usable transcript."""


class TranscriptSegment(BaseModel):
    """One timed chunk of speech, exactly as YouTube emitted it."""

    start: float
    duration: float
    text: str


class Transcript(BaseModel):
    """Every segment of one video, plus what it took to get them."""

    video_id: str
    language_code: str
    is_generated: bool
    segments: list[TranscriptSegment]


def parse_video_id(url: str) -> str:
    """Extract the 11-character video id from a YouTube URL.

    Args:
        url: A `youtu.be/<id>` or `youtube.com/watch?v=<id>` URL.

    Raises:
        TranscriptError: If no video id is present.
    """
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/")
    else:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    if not video_id:
        raise TranscriptError(f"No YouTube video id found in {url!r}.")
    return video_id


def fetch_transcript(url: str) -> Transcript:
    """Fetch the transcript for a YouTube URL, timings intact.

    Args:
        url: A YouTube video URL.

    Raises:
        TranscriptError: If the video has no retrievable transcript.
    """
    video_id = parse_video_id(url)
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id)
    except Exception as exc:
        raise TranscriptError(f"No transcript available for {video_id}: {exc}") from exc
    return Transcript(
        video_id=video_id,
        language_code=fetched.language_code,
        is_generated=fetched.is_generated,
        segments=[
            TranscriptSegment(start=s.start, duration=s.duration, text=s.text) for s in fetched
        ],
    )


def source_ref(video_id: str, segment: TranscriptSegment) -> str:
    """Build the citation anchor stored in `LearningEvent.source_ref`.

    This string is the only bridge between a concept the learner studied and the
    exact moment in the video it came from. It is written into an append-only
    event log, so its format is effectively permanent.

    A resolvable URL is stored rather than an internal id, because the first
    consumer of this log is a human checking whether a generated claim is
    honest. Making that check one click away is worth more than the sub-second
    precision lost to `?t=`, which takes whole seconds only. Truncating rounds
    *backwards*, so the link lands just before the cited words, not after them.

    Args:
        video_id: The 11-character YouTube video id.
        segment: The segment being cited.

    Returns:
        A YouTube URL that opens at the moment the segment starts.
    """
    return f"https://youtu.be/{video_id}?t={int(segment.start)}"
