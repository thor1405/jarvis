#!/usr/bin/env python3
import os
import sys
import socket
import threading
import psutil
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Static, Input, RichLog, Button
from textual.reactive import reactive
from textual.binding import Binding

SOCKET_PATH = "/tmp/jarvis.sock"

class TelemetryWidget(Static):
    cpu = reactive(0.0)
    ram = reactive(0.0)

    def on_mount(self):
        self.set_interval(1.0, self.update_stats)

    def update_stats(self):
        self.cpu = psutil.cpu_percent()
        self.ram = psutil.virtual_memory().percent
        self.update(
            f"[bold cyan]◈ NEURAL TELEMETRY ◈[/bold cyan]\n"
            f"[green]CPU LOAD :[/green] {self.cpu:>5.1f}% [{'#' * int(self.cpu / 10):<10}]\n"
            f"[green]RAM LOAD :[/green] {self.ram:>5.1f}% [{'#' * int(self.ram / 10):<10}]\n"
            f"[yellow]SHORTCUTS:[/yellow] [bold red]F2 (PTT)[/bold red] | [bold yellow]F3/ESC (Skip Speech)[/bold yellow]"
        )

class TacticalHUD(Static):
    state = reactive("STANDBY")

    def update_state(self, new_state: str):
        colors = {
            "STANDBY": "cyan", "RECORDING": "bold red",
            "THINKING": "bold yellow", "SPEAKING": "bold green"
        }
        col = colors.get(new_state, "white")
        self.update(
            f"[bold {col}]╔════════════════════════════════════╗\n"
            f"║   CORE STATE: {new_state:<20} ║\n"
            f"╚════════════════════════════════════╝[/bold {col}]"
        )

class JarvisTUI(App):
    BINDINGS = [
        Binding("f2", "toggle_ptt", "Record (F2)"),
        Binding("f3", "skip_speech", "Skip Speech (F3)"),
        Binding("escape", "skip_speech", "Skip Speech (ESC)")
    ]

    CSS = """
    Screen { background: #060913; color: #d1e8ff; }
    #top-row { height: 8; margin: 1; }
    #telemetry { border: heavy cyan; width: 48%; height: 100%; padding: 0 1; }
    #hud { border: heavy green; width: 48%; height: 100%; align: center middle; }
    #logs { border: heavy #1f3a60; height: 1fr; margin: 0 1; background: #02040a; }
    #bottom-bar { height: 4; margin: 1; }
    #cmd-input { width: 80%; border: tall cyan; background: #0b1329; }
    #skip-btn { width: 20%; height: 100%; border: tall red; background: #3b0d0d; color: white; text-style: bold; }
    #skip-btn:hover { background: #a81313; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top-row"):
            yield TelemetryWidget(id="telemetry")
            yield TacticalHUD(id="hud")
        yield RichLog(id="logs", highlight=True, markup=True)
        with Horizontal(id="bottom-bar"):
            yield Input(placeholder="Type command or press F2 for voice recording...", id="cmd-input")
            yield Button("⏹ SKIP SPEECH", id="skip-btn", variant="error")
        yield Footer()

    def on_mount(self):
        self.log_widget = self.query_one("#logs", RichLog)
        self.hud_widget = self.query_one("#hud", TacticalHUD)
        self.sock = None
        self.is_recording = False
        self.log_widget.write("[bold green]STARK INDUSTRIAL // JARVIS NEURAL INTERFACE ONLINE[/bold green]")
        threading.Thread(target=self.socket_listener_thread, daemon=True).start()

    def socket_listener_thread(self):
        while True:
            try:
                if os.path.exists(SOCKET_PATH):
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(SOCKET_PATH)
                    self.sock = s
                    self.app.call_from_thread(self.log_widget.write, "[green]Connected to Jarvis Backend Core.[/green]")
                    buffer = ""
                    while True:
                        data = s.recv(1024)
                        if not data:
                            break
                        buffer += data.decode("utf-8")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if "|" in line:
                                evt, val = line.split("|", 1)
                                if evt == "STATE":
                                    self.app.call_from_thread(self.hud_widget.update_state, val)
                                elif evt == "LOG":
                                    ts = datetime.now().strftime("%H:%M:%S")
                                    self.app.call_from_thread(self.log_widget.write, f"[grey]{ts}[/grey] » {val}")
                else:
                    self.sock = None
            except Exception:
                self.sock = None
            self.app.call_from_thread(self.hud_widget.update_state, "DISCONNECTED")
            threading.Event().wait(1.5)

    def action_skip_speech(self):
        """Silences speech output."""
        if self.sock:
            try:
                self.sock.sendall(b"SKIP_SPEECH\n")
            except Exception as e:
                self.log_widget.write(f"[red]Error sending skip: {e}[/red]")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "skip-btn":
            self.action_skip_speech()

    def action_toggle_ptt(self):
        if not self.sock:
            return
        if not self.is_recording:
            self.is_recording = True
            self.sock.sendall(b"START_PTT\n")
        else:
            self.is_recording = False
            self.sock.sendall(b"STOP_PTT\n")

    def on_input_submitted(self, message: Input.Submitted):
        cmd = message.value.strip()
        if not cmd:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_widget.write(f"[bold yellow]{ts} [USER][/bold yellow] » {cmd}")
        if self.sock:
            try:
                self.sock.sendall((cmd + "\n").encode("utf-8"))
            except Exception as e:
                self.log_widget.write(f"[red]Transmitting error: {e}[/red]")
        message.input.value = ""

if __name__ == "__main__":
    app = JarvisTUI()
    app.run()
