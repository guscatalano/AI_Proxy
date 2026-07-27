"""Telling a complete image from one that merely starts like one.

Magic bytes prove the first few bytes and nothing else. A screenshot sliced by a client's
tool-output cap keeps a perfect header — right format, right dimensions — and loses the pixels,
so it passed every check the proxy had and then rendered as a broken thumbnail with no
explanation. Observed in production: the same screenshot stored at 472,247 bytes when sent as a
vision part and 14,288 bytes when embedded in tool text that a 50,000-character cap cut in half.
"""
import base64
import json
import struct
import zlib

from ai_proxy import proxy as P


def _png(width=48, height=48, truncate_at=None):
    """A real PNG, comfortably past the 128-byte floor in _decode_b64_image.

    Pixels are noisy on purpose so zlib can't squeeze the file back under that floor, and
    nothing is appended after IEND — a real encoder ends there, and trailing bytes would make
    the terminator check meaningless.
    """
    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + bytes((x * 7 + y * 31) % 256 for x in range(width * 3))
                    for y in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b""))[:truncate_at]


def test_complete_png_is_complete():
    assert P._image_is_complete(_png()) is True


def test_png_missing_its_end_marker_is_not():
    whole = _png()
    assert P._image_is_complete(whole[:len(whole) // 3]) is False


def test_jpeg_terminator():
    body = b"\xff\xd8\xff" + b"\x11" * 400
    assert P._image_is_complete(body) is False
    assert P._image_is_complete(body + b"\xff\xd9") is True


def test_gif_terminator():
    body = b"GIF89a" + b"\x22" * 400
    assert P._image_is_complete(body) is False
    assert P._image_is_complete(body + b"\x3b") is True


def test_webp_uses_its_declared_length():
    payload = b"WEBPVP8 " + b"\x00" * 200
    riff = b"RIFF" + struct.pack("<I", len(payload)) + payload
    assert P._image_is_complete(riff) is True
    assert P._image_is_complete(riff[:-40]) is False


def test_unknown_formats_are_not_called_broken():
    # BMP has no terminator. Claiming truncation we cannot detect would be a false alarm.
    assert P._image_is_complete(b"BM" + b"\x00" * 400) is True


def test_a_truncated_image_still_passes_the_header_check():
    # The reason this was invisible: header validation says yes to both.
    cut = _png(truncate_at=300)
    b64 = base64.b64encode(cut).decode()
    assert P._decode_b64_image(b64) is not None      # header check: fine
    assert P._image_is_complete(cut) is False        # completeness: not fine


def _store(rid, png, viewer_ip="testclient"):
    conn = P.db()
    # The image endpoint applies the same visibility gate as the request detail, so the row has
    # to look like it came from the address the TestClient presents.
    conn.execute("INSERT OR REPLACE INTO requests (id, ts, method, path, upstream_url, "
                 "client_ip, has_images) VALUES (?, 1, 'POST', '/v1/messages', 'http://x', "
                 "?, 1)", (rid, viewer_ip))
    conn.execute("INSERT OR REPLACE INTO request_blobs (id, request_body, images_data) "
                 "VALUES (?, ?, ?)",
                 (rid, "{}", json.dumps([{"media_type": "image/png",
                                          "data": base64.b64encode(png).decode()}])))
    conn.commit()
    conn.close()


def test_detail_flags_a_truncated_image(client):
    _store("img-cut", _png(truncate_at=200))
    _store("img-ok", _png())
    cut = client.get("/__proxy/api/requests/img-cut").json()["images"][0]
    ok = client.get("/__proxy/api/requests/img-ok").json()["images"][0]
    assert cut["complete"] is False
    assert ok["complete"] is True


def test_image_endpoint_headers_say_which(client):
    _store("img-cut2", _png(truncate_at=200))
    _store("img-ok2", _png())
    r_cut = client.get("/__proxy/api/requests/img-cut2/image/0")
    r_ok = client.get("/__proxy/api/requests/img-ok2/image/0")
    assert r_cut.status_code == 200 and r_cut.headers["x-image-complete"] == "0"
    assert r_ok.status_code == 200 and r_ok.headers["x-image-complete"] == "1"
    # Still served: a partial screenshot is worth looking at, it just needs labelling.
    assert r_cut.content.startswith(b"\x89PNG")


def test_capture_regex_keeps_a_whole_payload(client):
    # The capture path was the first suspect and is not at fault: a well-formed blob embedded in
    # tool-result JSON comes back byte for byte. What arrives truncated, arrives truncated.
    b64 = base64.b64encode(_png(width=64, height=64)).decode()
    msg = json.dumps({"tool": "screenshot", "screenshot_png_b64": b64})
    runs = list(P._embedded_b64_images(msg))
    assert len(runs) == 1
    assert runs[0][1] == b64
