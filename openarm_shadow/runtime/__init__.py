"""Chạy teleop: luồng perception (worker), luồng điều khiển + về tư thế nghỉ (controller), tự engage (session),
ghi --record (recorder), vòng chính + phím (app)."""
from .app import open_perception, run
