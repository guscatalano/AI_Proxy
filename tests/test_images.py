"""Image detection: a request is only 'has_images' when there is an image you could open.

Regression suite for a bug that flagged 271 of 2353 requests/day from a text-only coding agent
as vision requests. Detection was a bare substring test — `"/9j/" in body_text` — and "/9j/" is
four base64 characters that occur by chance in paths, prose and unrelated blobs. Every resulting
badge led to an image that failed to render, because nothing validated that the bytes decoded.
"""
import base64

import ai_proxy.proxy as p


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


PNG = _b64(bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 400)
JPEG = _b64(b"\xff\xd8\xff" + b"\x00" * 400)
GIF = _b64(b"GIF89a" + b"\x00" * 400)


def test_accepts_real_images():
    for payload in (PNG, JPEG, GIF):
        assert p._decode_b64_image(payload) is not None


def test_rejects_a_stray_jpeg_marker_in_ordinary_text():
    """'/9j/' is 4 base64 chars. It appears inside file paths and long unrelated blobs, and was
    the main source of false vision flags."""
    assert p._decode_b64_image("some/9j/path/to/a/file/long/enough/to/clear/the/length/gate") is None
    assert p._decode_b64_image("/9j/" + "notbase64!!" * 30) is None


def test_rejects_truncated_payloads():
    """The production symptom: a fixed 1833-char run — not a multiple of 4 — stored for hundreds
    of requests, none of which could be decoded or displayed."""
    assert p._decode_b64_image("iVBORw0KGgo" + "A" * 1822) is None


def test_rejects_valid_base64_that_is_not_an_image():
    assert p._decode_b64_image(_b64(b"hello world" * 40)) is None


def test_rejects_payloads_too_small_to_be_an_image():
    """Magic bytes are 2-8 bytes long; matching them in a handful of decoded bytes is a
    coincidence, not a picture."""
    assert p._decode_b64_image(_b64(b"\x89PNG\r\n\x1a\n")) is None
    assert p._decode_b64_image(PNG[:40]) is None


def test_tolerates_wrapped_base64():
    """Data URLs are sometimes line-wrapped; whitespace must not defeat decoding."""
    wrapped = "\n".join(PNG[i:i + 76] for i in range(0, len(PNG), 76))
    assert p._decode_b64_image(wrapped) is not None


def test_rejects_non_string_input():
    for bad in (None, 123, {"data": PNG}, b"bytes"):
        assert p._decode_b64_image(bad) is None


def test_embedded_scan_finds_a_real_blob_in_tool_text():
    """The embedded-blob path (a screenshot returned inside a tool's text output) still works —
    tightening detection must not stop surfacing genuine images."""
    text = 'tool said: {"screenshot_png_b64": "' + PNG + '"} done'
    found = [b64 for _mt, b64 in p._embedded_b64_images(text) if p._decode_b64_image(b64)]
    assert found, "a genuine embedded PNG should still be detected"


def test_body_has_images_still_keys_off_structured_parts_only():
    """Vision ROUTING must key off declared image parts, never off text that happens to contain
    base64 — routing a text request to a vision model on a false positive breaks the request."""
    body = {"messages": [{"role": "user", "content": "here is /9j/ in my prose"}]}
    assert p._body_has_images(body) is False
    real = {"messages": [{"role": "user", "content": [{"type": "image_url",
                                                       "image_url": {"url": "data:image/png;base64," + PNG}}]}]}
    assert p._body_has_images(real) is True
