# server/usb_mcp_server.py
from mcp.server.fastmcp import FastMCP
from usb_utils import scan_usb_devices

mcp = FastMCP("usb-mcp",instructions="Detects connected devices and take actions with them."
)

@mcp.tool()
def list_usb(force_rescan: bool = False):
    """
    Return descrption of all connected devices.

    """
    devices = scan_usb_devices()
    return {"count": len(devices), "devices": devices}

if __name__ == "__main__":
    # stdio transport for Claude Desktop
    mcp.run()