import os, io, base64, json, logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
from PIL import Image
from anthropic import Anthropic, APIError
from mcp.server.fastmcp import FastMCP
import numpy as np


# ---- MCP app setup ----
mcp = FastMCP("Camera-MCP",instructions="interact with the connected camera. Anything that involves visual would need this. If there is only one camera attached use single_camera tool. If there are more than one camera, use multi_camera tool.")

# ---- Config via environment ----
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")

MAX_EDGE = 1024  # downscale to keep request light & snappy

# ---- helpers ----
def _resize_if_needed(pil_img: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    w, h = pil_img.size
    m = max(w, h)
    if m <= max_edge:
        return pil_img
    scale = max_edge / float(m)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return pil_img.resize(new_size, Image.LANCZOS)

def _capture_single_frame(device_index: int = 0, width: int = 0, height: int = 0) -> Image.Image:
    # OpenCV webcam capture (single frame)
    cap = cv2.VideoCapture(device_index)
    try:
        if width > 0 and height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera.")
        # Convert BGR (OpenCV) -> RGB (Pillow)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    finally:
        cap.release()

def _pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _build_prompt(purpose: str) -> str:
    # Keep it classification-only (no identity/biometrics), ask for strict JSON.
    return (
        "You will be given a photo. \n"
        "You are here to complete a task or answer a question. Right now it is to " + purpose + ".\n"
        "Report the answer of the purpose as JSON with this schema:\n"
        "{\n"
        '  "label": answer to previously stated question or task,\n'
        '  "confidence": 0.0-1.0,\n'
        '  "notes": "very short rationale (<= 20 words)"\n'
        "}\n"
        "Rules: Do NOT identify the person. If face is unclear or multiple faces exist, set label='other' and explain in notes."
    )

def _ask_claude(image_b64_jpeg: str, model: str, purpose: str) -> Dict[str, Any]:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    # Per Anthropic Vision docs: send an image block with base64 source and a text block with instructions.
    # https://docs.anthropic.com/claude/docs/vision
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64_jpeg
                        }
                    },
                    {
                        "type": "text",
                        "text": _build_prompt(purpose)
                    }
                ]
            }]
        )
    except APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}")

    # Extract text
    parts = []
    for block in getattr(msg, "content", []) or []:
        if block.type == "text":
            parts.append(block.text)
    text = "\n".join(parts).strip()
    # Try to parse JSON from model output
    # Be tolerant: find first { ... } block if extra text is present.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        maybe = text[start:end+1]
        try:
            return json.loads(maybe)
        except json.JSONDecodeError:
            pass
    return {"raw": text}

# ---- Tool: analyze_expression ----
@mcp.tool()
def single_camera(
    device_index: int = 0,
    width: int = 0,
    height: int = 0,
    purpose: str =''
) -> str:
    """
    Use ONLY when exactly one camera is available.

    Args:
        device_index: OpenCV camera index (default 0).
        width: Optional capture width; 0 keeps camera default.
        height: Optional capture height; 0 keeps camera default.
        purpose: A string describing the intended purpose or task for the captured image.

        Returns:
            A JSON string containing the fields: label, confidence, scores{...}, and notes.
            If JSON parsing fails, returns {"raw": "..."} with the raw text response.
    """
    if not ANTHROPIC_API_KEY:
        return json.dumps({"error": "Set ANTHROPIC_API_KEY in your environment."})

    img = _capture_single_frame(device_index, width, height)
    img = _resize_if_needed(img, MAX_EDGE)
    cv2.imshow("Captured Frame", np.array(img))  # show the window
    cv2.waitKey(0)                       # wait for key press (0 = indefinitely)
    cv2.destroyAllWindows() 
    b64 = _pil_to_base64_jpeg(img, quality=90)

    result = _ask_claude(b64, ANTHROPIC_MODEL, purpose)
    return json.dumps(result, ensure_ascii=False)

@mcp.tool()
def multi_camera(
    device_indices: str = "",      # e.g. "0,1"
    widths: str = "",              # e.g. "640,640"  (optional)
    heights: str = "",             # e.g. "480,480"  (optional)
    purpose: str = ""              # description of what Claude should do with both images
) -> str:
    """
    Use ONLY when two or more camera indices are provided (e.g., "0,1").
    Sends ALL frames together to Claude and returns ONE combined result.

    Args:
        device_indices: Comma-separated camera indices (e.g., "0,1").
                        If empty, defaults to [0].

        widths:         Comma-separated list of capture widths for each camera.
                        If fewer values are given, remaining cameras use default width.

        heights:        Comma-separated list of capture heights for each camera.
                        If fewer values are given, remaining cameras use default height.

        purpose:        A string describing the overall task or analysis purpose
                        for all captured images.

    Returns:
        A JSON string containing:
        {
            "combined_response": {...parsed Claude result...},
            "errors": [
                { "camera_index": int, "error": "message" }
            ]
        }

        If parsing fails, returns {"raw": "..."} with the unparsed text.
    """
    # ---------------------------------------------------------
    # Environment check
    # ---------------------------------------------------------
    if not ANTHROPIC_API_KEY:
        return json.dumps({"error": "Set ANTHROPIC_API_KEY in your environment."})

    # ---------------------------------------------------------
    # Parse inputs
    # ---------------------------------------------------------
    try:
        cam_indices = [int(x.strip()) for x in device_indices.split(",") if x.strip()] or [0]
        widths_list = [int(x.strip()) for x in widths.split(",") if x.strip()]
        heights_list = [int(x.strip()) for x in heights.split(",") if x.strip()]
    except ValueError:
        return json.dumps({"error": "Invalid format: indices, widths, and heights must be integers."})

    # ---------------------------------------------------------
    # Capture frames
    # ---------------------------------------------------------
    images = []
    errors = []

    for i, cam_idx in enumerate(cam_indices):
        try:
            w = widths_list[i] if i < len(widths_list) else 0
            h = heights_list[i] if i < len(heights_list) else 0

            img = _capture_single_frame(cam_idx, w, h)
            img = _resize_if_needed(img, MAX_EDGE)
            b64 = _pil_to_base64_jpeg(img, quality=90)

            images.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64
                }
            })

        except Exception as e:
            errors.append({
                "camera_index": cam_idx,
                "error": str(e)
            })

    if not images:
        return json.dumps({"error": "No valid frames captured."})

    # ---------------------------------------------------------
    # Build multi-image request to Claude
    # ---------------------------------------------------------
    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": images + [{
                    "type": "text",
                    "text": (
                        "You are given multiple images captured from different cameras.\n"
                        f"Your task: {purpose}\n"
                        "Use all images together to form one coherent response."
                    )
                }]
            }]
        )

        # Extract text
        parts = []
        for block in getattr(msg, "content", []) or []:
            if block.type == "text":
                parts.append(block.text)
        text = "\n".join(parts).strip()

        # Try to parse JSON if possible
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            maybe = text[start:end+1]
            try:
                parsed = json.loads(maybe)
                return json.dumps({
                    "combined_response": parsed,
                    "errors": errors
                }, ensure_ascii=False)
            except json.JSONDecodeError:
                pass

        return json.dumps({
            "combined_response": {"raw": text},
            "errors": errors
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "errors": errors
        }, ensure_ascii=False)

def main():
    mcp.run()

if __name__ == "__main__":
    main()