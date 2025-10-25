import serial
import serial.tools.list_ports

def scan_serial_devices():
    ports = serial.tools.list_ports.comports()
    return [port.__dict__ for port in ports]