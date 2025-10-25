import libusb_package
import usb.core
import usb.util

def _safe_str(dev, idx):
    if not idx:
        return None
    try:
        return usb.util.get_string(dev, idx)
    except Exception:
        return None
    
def scan_usb_devices():
    """
    Returns a list of devices with common fields:
    [
      {
        "vid": "046d", "pid": "c534",
        "manufacturer": "Logitech", "product": "USB Receiver",
        "serial": "ABC123" | None,
        "bDeviceClass": 0, "bDeviceSubClass": 0, "bDeviceProtocol": 0
      }, ...
    ]
    """
    devices = []
    for dev in libusb_package.find(find_all=True):
        manufacturer = _safe_str(dev, dev.iManufacturer) or "Unknown"
        product      = _safe_str(dev, dev.iProduct)      or "Unknown"
        serial       = _safe_str(dev, dev.iSerialNumber)
        devices.append({
            "vid": f"{dev.idVendor:04x}",
            "pid": f"{dev.idProduct:04x}",
            "manufacturer": manufacturer,
            "product": product,
            "serial": serial,
            "bDeviceClass": getattr(dev, "bDeviceClass", None),
            "bDeviceSubClass": getattr(dev, "bDeviceSubClass", None),
            "bDeviceProtocol": getattr(dev, "bDeviceProtocol", None),
        })
    return devices