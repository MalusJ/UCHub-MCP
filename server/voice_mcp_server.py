from __future__ import annotations
import base64, io, os, tempfile
import sys
from typing import Optional, Dict, Any
import requests
from pydub import AudioSegment  # requires ffmpeg on PATH
from elevenlabs import ElevenLabs
from mcp.server.fastmcp import FastMCP
import voice_util

mcp = FastMCP("voice-mcp",instructions="interact with the conencted mic. If the usb mcp server shows that there is a mic connected and I ask you to record. Use this voice mcp server to do that.")

@mcp.tool()
def voice_record(force_rescan: bool = False):
    """
    Start recording the mic. Return a message saying the mic feature is activated.

    """
    return "Mic is activated. You can now speak to the mic."
#listen for convert

# always listen
#

if __name__ == "__main__":
    # stdio transport for Claude Desktop
    mcp.run()



# how to call elevenlabs
# # You should have set up your API key as environment variable
# key = os.getenv("ELEVENLABS_API_KEY")
# print(key, file=sys.stderr)
# client = ElevenLabs(api_key=key)
# # change to absolute path method later"
# with open(r"C:\\Users\\fengy\\source\\repos\\MCP\\UCHub-MCP\\src\\voice_recording\\recording1.webm", "rb") as f:
#     transcription = client.speech_to_text.convert(
#         file=f,
#         model_id="scribe_v1"
#     )