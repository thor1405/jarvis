import os
import sys
import time
import re
import glob
import shutil
import wave
import signal
import socket
import threading
import subprocess
import urllib.parse
import contextlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudio
from faster_whisper import WhisperModel
import ollama

# ==========================================
# 0. SUPPRESS ALSA / BACKEND LOGS
# ==========================================
@contextlib.contextmanager
def no_alsa_err():
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        except Exception:
            pass

# ==========================================
# 1. HARDWARE & ENGINE CONFIG
# ==========================================
PIPER_MODEL = os.path.expanduser("~/jarvis_os/models/piper/en_US-lessac-medium.onnx")
PIPER_CONFIG = os.path.expanduser("~/jarvis_os/models/piper/en_US-lessac-medium.onnx.json")
WHISPER_SIZE = "base.en"
LLM_MODEL = "qwen2.5:7b"

RATE = 48000
CHANNELS = 2
CHUNK_SIZE = 2048

IS_SPEAKING = False
CURRENT_PROC = None
PROC_LOCK = threading.Lock()

RECORDING_ACTIVE = False
AUDIO_FRAMES = []

SOCKET_PATH = "/tmp/jarvis.sock"
CONNECTED_CLIENTS = []
CLIENT_LOCK = threading.Lock()

print("[Init] Loading Whisper speech engine...")
with no_alsa_err():
    stt_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

# ==========================================
# 2. STATUS BROADCASTER
# ==========================================
def emit_status(status_type: str, message: str):
    payload = f"{status_type}|{message}\n".encode("utf-8")
    with CLIENT_LOCK:
        dead = []
        for client in CONNECTED_CLIENTS:
            try:
                client.sendall(payload)
            except Exception:
                dead.append(client)
        for d in dead:
            if d in CONNECTED_CLIENTS:
                CONNECTED_CLIENTS.remove(d)

# ==========================================
# 3. ROBUST AUDIO DSP & INSTANT SKIP
# ==========================================
def convert_to_16k_mono(raw_bytes: bytes) -> bytes:
    audio = np.frombuffer(raw_bytes, dtype=np.int16)
    mono = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return mono[::3].tobytes()

def stop_playback():
    """Instantly kills Piper and audio playback."""
    global CURRENT_PROC, IS_SPEAKING
    with PROC_LOCK:
        if CURRENT_PROC is not None:
            try:
                os.killpg(os.getpgid(CURRENT_PROC.pid), signal.SIGKILL)
            except Exception:
                pass
            CURRENT_PROC = None
    subprocess.run(["pkill", "-9", "-f", "aplay -r 22050"], capture_output=True)
    IS_SPEAKING = False
    emit_status("STATE", "STANDBY")
    emit_status("LOG", "Speech skipped.")

def play_single_sentence(sentence: str) -> bool:
    global CURRENT_PROC, IS_SPEAKING
    clean = sentence.replace('"', '\\"').replace("'", "").replace("\n", " ").strip()
    if not clean:
        return True

    IS_SPEAKING = True
    emit_status("STATE", "SPEAKING")
    emit_status("LOG", f"Speaking: {clean}")

    piper_cmd = f'echo "{clean}" | piper --model {PIPER_MODEL} --config {PIPER_CONFIG} --output-raw | aplay -r 22050 -f S16_LE -t raw -q'
    proc = subprocess.Popen(piper_cmd, shell=True, preexec_fn=os.setsid)
    with PROC_LOCK:
        CURRENT_PROC = proc

    proc.wait()
    with PROC_LOCK:
        CURRENT_PROC = None

    time.sleep(0.06)
    IS_SPEAKING = False
    emit_status("STATE", "STANDBY")
    return True

# ==========================================
# 4. ALL FUNCTION DEFINITIONS (ORDERED FIRST)
# ==========================================
def get_desktop_env():
    env = os.environ.copy()
    if "DISPLAY" not in env:
        env["DISPLAY"] = ":0"
    if "WAYLAND_DISPLAY" not in env:
        for candidate in ["wayland-0", "wayland-1"]:
            if os.path.exists(f"/run/user/{os.getuid()}/{candidate}"):
                env["WAYLAND_DISPLAY"] = candidate
                break
    if "XDG_RUNTIME_DIR" not in env:
        env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
    if "DBUS_SESSION_BUS_ADDRESS" not in env:
        bus = f"/run/user/{os.getuid()}/bus"
        if os.path.exists(bus):
            env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus}"
    return env

def resolve_system_path(raw_path: str) -> Path:
    cleaned = raw_path.strip().strip("'\"")
    expanded = Path(os.path.expanduser(cleaned))
    if expanded.is_absolute() and expanded.exists():
        return expanded.resolve()

    common = {
        "downloads": Path.home() / "Downloads", "documents": Path.home() / "Documents",
        "desktop": Path.home() / "Desktop", "pictures": Path.home() / "Pictures",
        "screenshots": Path.home() / "Pictures/Screenshots",
        "videos": Path.home() / "Videos", "music": Path.home() / "Music", "home": Path.home()
    }
    low = cleaned.lower().rstrip("/\\")
    if low in common:
        return common[low]

    for name, p in common.items():
        if low.startswith(name + "/"):
            return (p / cleaned[len(name) + 1:]).resolve()

    for base in [Path.home() / "Pictures", Path.home() / "Documents", Path.home() / "Downloads", Path.home() / "Desktop"]:
        candidate = base / cleaned
        if candidate.exists():
            return candidate.resolve()
        for sub in base.iterdir():
            if sub.is_dir() and sub.name.lower() == low:
                return sub.resolve()

    return expanded.resolve()

def launch_app(app_name: str):
    """Launches desktop applications, terminals, or browsers cleanly under Wayland/systemd."""
    clean = app_name.lower().strip()
    env = get_desktop_env()

    if clean in ["terminal", "new terminal", "terminal window", "gnome terminal", "console"]:
        for term_bin in ["gnome-terminal --window", "ptyxis --new-window", "alacritty", "kitty", "x-terminal-emulator"]:
            bin_name = term_bin.split()[0]
            if shutil.which(bin_name):
                subprocess.Popen(term_bin, shell=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                emit_status("LOG", "Opened Terminal")
                return "Opened a new terminal window."

    if clean in ["text editor", "editor", "gedit", "gnome text editor"]:
        for ed in ["gnome-text-editor", "gedit", "kate"]:
            if shutil.which(ed):
                subprocess.Popen([ed], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                emit_status("LOG", "Opened Text Editor")
                return "Opened the text editor."

    if clean in ["firefox", "browser", "chrome", "google chrome"]:
        for br in (["firefox"] if "firefox" in clean else ["google-chrome", "chromium", "firefox", "brave-browser"]):
            if shutil.which(br):
                subprocess.Popen([br], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                emit_status("LOG", f"Opened {br.title()}")
                return f"Opened {br.title()}."

    if clean in ["files", "file manager"]:
        return open_folder("home")

    if clean in ["trash", "trash bin", "rubbish"]:
        return open_trash()

    app_map = {
        "app center": ["snap-store", "ubuntu-app-center", "gnome-software"],
        "calculator": ["gnome-calculator"], "settings": ["gnome-control-center"],
        "code": ["code"]
    }
    candidates = app_map.get(clean, [clean])
    for c in candidates:
        if subprocess.run(["gtk-launch", c], env=env, capture_output=True).returncode == 0:
            emit_status("LOG", f"Launched: {app_name}")
            return f"Opened {app_name}."
        if shutil.which(c):
            subprocess.Popen([c], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            emit_status("LOG", f"Launched: {app_name}")
            return f"Opened {app_name}."

    return f"Could not find application '{app_name}'."

def set_system_volume(percent: int = None, delta: int = None):
    try:
        subprocess.run("wpctl set-mute @DEFAULT_AUDIO_SINK@ 0 2>/dev/null || pactl set-sink-mute @DEFAULT_SINK@ 0 2>/dev/null", shell=True)
        if delta is not None:
            sign = "+" if delta > 0 else "-"
            subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{abs(delta)}%{sign}"], capture_output=True)
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{delta:+d}%"], capture_output=True)
            emit_status("LOG", f"Volume: {delta:+d}%")
            return f"Volume adjusted by {delta} percent."
        if percent is not None:
            pct = max(0, min(100, int(percent)))
            subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", str(round(pct / 100.0, 2))], capture_output=True)
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"], capture_output=True)
            emit_status("LOG", f"Volume: {pct}%")
            return f"Volume set to {pct} percent."
    except Exception as e:
        return f"Failed to change volume: {str(e)}"
    return "Volume command not understood."

def set_system_brightness(percent: int = None, delta: int = None):
    emit_status("LOG", "Adjusting screen brightness...")
    try:
        if not shutil.which("brightnessctl"):
            return "brightnessctl is not installed. Please run: sudo apt install brightnessctl"
        if delta is not None:
            sign = "+" if delta > 0 else "-"
            subprocess.run(["brightnessctl", "set", f"{abs(delta)}%{sign}"], capture_output=True)
            emit_status("LOG", f"Brightness: {delta:+d}%")
            return f"Brightness adjusted by {delta} percent."
        if percent is not None:
            pct = max(1, min(100, int(percent)))
            subprocess.run(["brightnessctl", "set", f"{pct}%"], capture_output=True)
            emit_status("LOG", f"Brightness: {pct}%")
            return f"Brightness set to {pct} percent."
    except Exception as e:
        return f"Failed to set brightness: {str(e)}"
    return "Brightness command not understood."

def get_current_time_and_date():
    now = datetime.now()
    time_str = now.strftime("%I:%M %p").lstrip('0')
    date_str = now.strftime("%A, %B %d, %Y")
    emit_status("LOG", f"Time checked: {time_str}")
    return f"The current time is {time_str}, and today is {date_str}."

def open_folder(folder_name: str = "home", subfolder_of: str = None):
    clean = folder_name.lower().strip() if folder_name else "home"
    env = get_desktop_env()

    if subfolder_of:
        parent = resolve_system_path(subfolder_of)
        target = parent / clean
    else:
        target = resolve_system_path(clean)

    if not target.exists():
        if "new folder" in clean:
            target.mkdir(parents=True, exist_ok=True)
        else:
            return f"Folder '{clean}' not found."

    subprocess.Popen(["xdg-open", str(target)], env=env)
    emit_status("LOG", f"Opened folder: {target.name}")
    return f"Opened {target.name} folder."

def open_file_anywhere(filename: str, folder: str = None):
    clean_name = filename.strip().strip("'\"")
    env = get_desktop_env()
    emit_status("LOG", f"Locating and opening '{clean_name}'...")

    target_path = None
    if folder:
        base = resolve_system_path(folder)
        candidate = base / clean_name
        if candidate.exists():
            target_path = candidate

    if not target_path:
        for search_base in [Path.home() / "Pictures/Screenshots", Path.home() / "Pictures",
                            Path.home() / "Downloads", Path.home() / "Documents",
                            Path.home() / "Desktop", Path.home()]:
            if not search_base.exists():
                continue
            matches = list(search_base.rglob(clean_name))
            if matches:
                target_path = matches[0]
                break

    if target_path and target_path.exists():
        subprocess.Popen(["xdg-open", str(target_path)], env=env)
        emit_status("LOG", f"Opened file: {target_path.name}")
        return f"Opened {target_path.name}."
    return f"Could not find '{clean_name}' to open."

def find_file_or_folder(target: str = "", min_size_mb: float = None, search_dir: str = "~"):
    clean = target.strip().strip("'\"") if target else ""
    base = resolve_system_path(search_dir)
    emit_status("LOG", f"Searching files {clean or f'> {min_size_mb}MB'}...")

    size_match = re.search(r'(\d+)\s*(mb|gb|kb)', clean.lower())
    if size_match and min_size_mb is None:
        val = float(size_match.group(1))
        unit = size_match.group(2)
        min_size_mb = val if unit == 'mb' else (val * 1024 if unit == 'gb' else val / 1024)
        clean = ""

    results = []
    threshold_bytes = (min_size_mb * 1024 * 1024) if min_size_mb else 0

    scan_dirs = [base] if base != Path.home() else [
        Path.home() / "Downloads", Path.home() / "Documents",
        Path.home() / "Videos", Path.home() / "Pictures",
        Path.home() / "Desktop", Path.home()
    ]

    for d in scan_dirs:
        if not d.exists():
            continue
        try:
            for p in d.rglob("*"):
                if p.is_file() and not any(part.startswith(".") for part in p.parts):
                    if clean and clean.lower() not in p.name.lower():
                        continue
                    if threshold_bytes > 0:
                        try:
                            if p.stat().st_size < threshold_bytes:
                                continue
                        except Exception:
                            continue
                    sz_mb = p.stat().st_size / (1024 * 1024)
                    results.append(f"{p.name} ({sz_mb:.1f} MB in {p.parent.name})")
                    if len(results) >= 8:
                        break
        except Exception:
            continue
        if len(results) >= 8:
            break

    if results:
        emit_status("LOG", f"Found {len(results)} matches.")
        return f"Found matching files: {', '.join(results)}."
    return "No matching files found."

def list_files_by_type_or_state(file_type: str = "all", directory: str = "~", include_hidden: bool = False):
    target_dir = resolve_system_path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        return f"Directory '{directory}' does not exist."

    type_exts = {
        "pdf": [".pdf"],
        "image": [".png", ".jpg", ".jpeg", ".webp", ".gif"],
        "images": [".png", ".jpg", ".jpeg", ".webp", ".gif"],
        "video": [".mp4", ".mkv", ".avi", ".mov"],
        "document": [".pdf", ".docx", ".txt", ".md", ".csv"],
        "all": []
    }
    valid_exts = type_exts.get(file_type.lower().strip(), [])

    items = []
    for p in target_dir.iterdir():
        if not include_hidden and p.name.startswith("."):
            continue
        if include_hidden and not p.name.startswith("."):
            continue
        if valid_exts and p.suffix.lower() not in valid_exts:
            continue
        items.append(p.name)

    if not items:
        category = "hidden" if include_hidden else file_type
        return f"No {category} files found in {target_dir.name}."

    summary = ", ".join(items[:10])
    count_extra = len(items) - 10
    if count_extra > 0:
        summary += f", and {count_extra} more"
    emit_status("LOG", f"Listed {len(items)} files.")
    return f"Files: {summary}."

def get_disk_usage_and_clean_advice():
    total, used, free = shutil.disk_usage(Path.home())
    free_gb = free // (2**30)
    total_gb = total // (2**30)

    cleanable = []
    cache_dir = Path.home() / ".cache"
    if cache_dir.exists():
        cleanable.append("User Cache (~/.cache)")

    down_dir = Path.home() / "Downloads"
    if down_dir.exists():
        old_installers = list(down_dir.glob("*.deb")) + list(down_dir.glob("*.iso")) + list(down_dir.glob("*.tar.gz"))
        if old_installers:
            cleanable.append(f"{len(old_installers)} installers in Downloads")

    trash_dir = Path.home() / ".local/share/Trash/files"
    if trash_dir.exists() and any(trash_dir.iterdir()):
        cleanable.append("Trash bin contents")

    advice = f"You have {free_gb} GB free out of {total_gb} GB. "
    if cleanable:
        advice += f"Recommended items to clean: {', '.join(cleanable)}."
    else:
        advice += "Your disk space is well optimized."
    emit_status("LOG", f"Disk: {free_gb}GB free")
    return advice

def empty_trash():
    env = get_desktop_env()
    try:
        if shutil.which("gio"):
            subprocess.run(["gio", "trash", "--empty"], env=env, capture_output=True)
            emit_status("LOG", "Trash emptied.")
            return "The trash has been emptied."
    except Exception:
        pass
    return "Trash emptied."

def open_trash():
    env = get_desktop_env()
    subprocess.Popen(["xdg-open", "trash://"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    emit_status("LOG", "Opened Trash.")
    return "Opened the trash bin."

def copy_file_or_folder(source: str, destination: str, filename: str = None):
    try:
        src = resolve_system_path(source)
        dest = resolve_system_path(destination)
        if filename:
            src = src / filename if src.is_dir() else src
        if not src.exists():
            for b in [Path.home() / "Documents", Path.home() / "Downloads", Path.home() / "Desktop"]:
                if (b / source).exists():
                    src = b / source
                    break
        dest.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(str(src), str(dest / src.name), dirs_exist_ok=True)
        else:
            shutil.copy2(str(src), str(dest / src.name if dest.is_dir() else dest))
        emit_status("LOG", f"Copied {src.name} -> {dest.name}")
        return f"Copied {src.name} to {dest.name}."
    except Exception as e:
        return f"Failed: {str(e)}"

def delete_file_or_folder(target: str, folder: str = None):
    try:
        raw = target.strip().strip("'\"")
        candidate = resolve_system_path(folder) / raw if folder else resolve_system_path(raw)
        if not candidate.exists():
            for b in [Path.home() / "Downloads", Path.home() / "Documents", Path.home() / "Desktop"]:
                if (b / raw).exists():
                    candidate = b / raw
                    break
        if candidate.exists():
            subprocess.run(["gio", "trash", str(candidate)], capture_output=True)
            emit_status("LOG", f"Deleted: {candidate.name}")
            return f"Deleted {candidate.name}."
    except Exception as e:
        return f"Failed: {str(e)}"
    return f"File '{target}' not found."

def control_window_state(app_name: str = None, action: str = "minimize"):
    clean_act = (action or "minimize").lower().strip()
    clean_app = (app_name or "").lower().strip()
    env = get_desktop_env()

    alias_map = {
        "browser": ["firefox", "chrome"], "firefox": ["firefox"], "chrome": ["google-chrome"],
        "terminal": ["gnome-terminal", "ptyxis"], "files": ["nautilus"]
    }
    win_id = None
    if shutil.which("xdotool"):
        for c in alias_map.get(clean_app, [clean_app]):
            try:
                out = subprocess.check_output(f"xdotool search --onlyvisible --class '{c}' 2>/dev/null", shell=True, env=env).decode().strip()
                if out:
                    win_id = out.splitlines()[-1]
                    break
            except Exception:
                continue

    if win_id and shutil.which("xdotool"):
        if clean_act in ["minimize", "minimise", "hide"]:
            subprocess.run(f"xdotool windowminimize {win_id}", shell=True, env=env)
        elif clean_act in ["maximize", "maximise"]:
            subprocess.run(f"xdotool windowactivate {win_id}; xdotool key alt+F10", shell=True, env=env)
        return f"{clean_act.title()}d {clean_app or 'window'}."

    if shutil.which("xdotool"):
        key = "Super+h" if "min" in clean_act else "Super+Up"
        subprocess.run(f"xdotool key {key}", shell=True, env=env)
        return f"{clean_act.title()}d active window."
    return "Window control unavailable."

def close_application(app_name: str):
    clean = app_name.lower().strip()
    alias_map = {"browser": ["firefox", "chrome"], "firefox": ["firefox"], "files": ["nautilus"]}
    for t in alias_map.get(clean, [clean]):
        subprocess.run(["pkill", "-15", "-f", t], capture_output=True)
    emit_status("LOG", f"Closed: {app_name}")
    return f"Closed {app_name}."

def web_search(query: str, platform: str = "auto"):
    q_clean = query.lower().strip()
    if q_clean in ["youtube", "open youtube"]:
        subprocess.Popen(["xdg-open", "https://www.youtube.com"], env=get_desktop_env())
        return "Opened YouTube."
    if q_clean in ["google", "open google"]:
        subprocess.Popen(["xdg-open", "https://www.google.com"], env=get_desktop_env())
        return "Opened Google."

    if "youtube" in platform.lower() or any(k in q_clean for k in ["video", "song", "play"]):
        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"
    else:
        url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
    subprocess.Popen(["xdg-open", url], env=get_desktop_env())
    emit_status("LOG", f"Searched: {query}")
    return f"Searching for '{query}'."

def generate_and_save_code(filename: str, language: str, description: str, folder: str = "Desktop"):
    target_folder = resolve_system_path(folder)
    target_folder.mkdir(parents=True, exist_ok=True)
    file_path = target_folder / filename

    emit_status("LOG", f"Generating {language} code for {filename}...")
    prompt = f"Write complete, production-ready {language} code for: {description}. Return ONLY the raw code without markdown backticks or conversational explanations."
    try:
        res = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
        raw_code = res.get("message", {}).get("content", "")
        cleaned_code = re.sub(r"^```[\w]*\n", "", raw_code, flags=re.MULTILINE)
        cleaned_code = re.sub(r"```$", "", cleaned_code, flags=re.MULTILINE).strip()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_code)

        emit_status("LOG", f"Saved: {file_path.name} in {target_folder.name}")
        return f"Generated {language} program and saved it as {filename} in {target_folder.name}."
    except Exception as e:
        return f"Failed to generate code: {str(e)}"

def execute_code_file(filename: str, folder: str = None, interactive: bool = True):
    file_path = None
    if folder:
        f_dir = resolve_system_path(folder)
        if (f_dir / filename).exists():
            file_path = f_dir / filename

    if not file_path:
        for sub in [Path.home() / "Desktop", Path.home() / "Documents", Path.home() / "Downloads", Path.home()]:
            if (sub / filename).exists():
                file_path = sub / filename
                break

    if not file_path or not file_path.exists():
        return f"File '{filename}' not found for execution."

    emit_status("LOG", f"Executing: {file_path.name}...")
    ext = file_path.suffix.lower()
    env = get_desktop_env()

    if ext == ".py":
        interp = "python3"
    elif ext == ".sh":
        interp = "bash"
    elif ext in [".js", ".mjs"]:
        interp = "node"
    elif ext == ".go":
        interp = "go run"
    else:
        interp = "xdg-open"

    needs_interactive = interactive
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        if any(kw in content for kw in ["input(", "readline(", "read -p", "prompt("]):
            needs_interactive = True
    except Exception:
        pass

    if needs_interactive:
        for term in ["gnome-terminal", "ptyxis", "alacritty", "kitty", "x-terminal-emulator"]:
            if shutil.which(term):
                if term == "gnome-terminal":
                    cmd = f'gnome-terminal --title="{file_path.name}" -- bash -c "{interp} \\"{file_path}\\"; echo; echo \\"[Process completed. Press Enter to close]\\"; read; exec bash"'
                else:
                    cmd = f'{term} -e bash -c "{interp} \\"{file_path}\\"; echo; echo \\"[Process completed. Press Enter to close]\\"; read"'
                subprocess.Popen(cmd, shell=True, env=env)
                emit_status("LOG", f"Launched {file_path.name} in terminal window.")
                return f"Opened and executed {file_path.name} in a new terminal window."

    try:
        run_cmd = [interp, str(file_path)] if " " not in interp else interp.split() + [str(file_path)]
        res = subprocess.run(run_cmd, capture_output=True, text=True, timeout=12)
        out = (res.stdout or res.stderr).strip()
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        result_preview = ", ".join(lines) if lines else "Program executed with no output."
        emit_status("LOG", f"Output: {result_preview}")
        return f"Execution finished. Output: {result_preview}"
    except subprocess.TimeoutExpired:
        return f"Program '{filename}' timed out after 12 seconds."
    except Exception as e:
        return f"Failed to execute '{filename}': {str(e)}"

# ==========================================
# 5. DISPATCHER & TOOL SCHEMAS
# ==========================================
AVAILABLE_TOOLS = {
    "launch_app": launch_app,
    "set_system_brightness": set_system_brightness,
    "set_system_volume": set_system_volume,
    "open_folder": open_folder,
    "open_file_anywhere": open_file_anywhere,
    "find_file_or_folder": find_file_or_folder,
    "list_files_by_type_or_state": list_files_by_type_or_state,
    "get_disk_usage_and_clean_advice": get_disk_usage_and_clean_advice,
    "get_current_time_and_date": get_current_time_and_date,
    "empty_trash": empty_trash,
    "open_trash": open_trash,
    "copy_file_or_folder": copy_file_or_folder,
    "delete_file_or_folder": delete_file_or_folder,
    "generate_and_save_code": generate_and_save_code,
    "execute_code_file": execute_code_file,
    "close_application": close_application,
    "control_window_state": control_window_state,
    "web_search": web_search
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "set_system_brightness",
            "description": "Adjusts screen brightness/backlight (e.g. 'increase the brightness to 100%', 'dim display by 20%').",
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {"type": "integer", "description": "Target percentage 1-100"},
                    "delta": {"type": "integer", "description": "Delta to increase or decrease"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_system_volume",
            "description": "Adjusts speaker audio volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {"type": "integer"},
                    "delta": {"type": "integer"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_folder",
            "description": "Opens folders, nested subfolders (e.g. 'screenshots', 'open new folder in documents').",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_name": {"type": "string", "description": "Target folder name, e.g. 'screenshots', 'new folder'"},
                    "subfolder_of": {"type": "string", "description": "Parent directory if specified, e.g. 'documents'"}
                },
                "required": ["folder_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_file_anywhere",
            "description": "Finds and opens any image, PNG, PDF, or document (e.g. 'open 1.png', 'open report.pdf').",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file, e.g. '1.png'"},
                    "folder": {"type": "string", "description": "Optional directory location, e.g. 'screenshots'"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_file_or_folder",
            "description": "Finds files matching names or large files exceeding size (e.g. 'find files greater than 100mb', 'find pass.txt').",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "min_size_mb": {"type": "number", "description": "Size threshold in megabytes"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files_by_type_or_state",
            "description": "Lists files by format or reveals hidden files (e.g. 'list all pdfs', 'list images', 'open hidden files', 'show hidden files').",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_type": {"type": "string", "enum": ["all", "pdf", "images", "video", "document"]},
                    "directory": {"type": "string"},
                    "include_hidden": {"type": "boolean"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_disk_usage_and_clean_advice",
            "description": "Shows disk space usage and suggests unnecessary files/caches to clean.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time_and_date",
            "description": "Gets current time and date.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "empty_trash",
            "description": "Empties trash bin.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_trash",
            "description": "Opens trash bin folder.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "launch_app",
            "description": "Launches apps (firefox, terminal, text editor, settings).",
            "parameters": {
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": ["app_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches Google or YouTube.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }
]

SYSTEM_PROMPT = (
    "You are Jarvis, an autonomous desktop assistant running natively on Linux. "
    "Rules:\n"
    "1. Respond in strictly 1 short sentence.\n"
    "2. GENERAL KNOWLEDGE: If the user asks a factual question (e.g. 'who is Charles Babbage', 'what is photosynthesis', 'why is the sky blue'), DO NOT search Google or open a browser. Directly explain it concisely in 1 sentence.\n"
    "3. NEVER output raw syntax like function_name(arg=val). Use tool_calls.\n"
    "4. BRIGHTNESS: For 'increase the brightness', 'set brightness to 100%', call set_system_brightness. Do NOT confuse brightness with volume.\n"
    "5. FOLDERS: For 'open screenshots folder', call open_folder(folder_name='screenshots'). For 'open new folder in documents', call open_folder(folder_name='new folder', subfolder_of='documents').\n"
    "6. OPENING FILES: For 'open 1.png', 'open sample.pdf', call open_file_anywhere.\n"
    "7. LISTING & SEARCH: For 'list pdfs', call list_files_by_type_or_state(file_type='pdf'). For 'open hidden files' or 'list hidden files', call list_files_by_type_or_state(include_hidden=True). For 'find files greater than 100mb', call find_file_or_folder with min_size_mb=100.\n"
    "8. DISK USAGE & CLEANUP: For 'show disk usage' or 'help clean space', call get_disk_usage_and_clean_advice."
)

def query_llm(prompt: str) -> str:
    emit_status("STATE", "THINKING")
    emit_status("LOG", f"Prompt: {prompt}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        response = ollama.chat(model=LLM_MODEL, messages=messages, tools=TOOL_SCHEMAS)
        msg = response.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if tool_calls:
            results = []
            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                emit_status("LOG", f"Tool: {name}")
                if name in AVAILABLE_TOOLS:
                    try:
                        res = AVAILABLE_TOOLS[name](**args)
                        if res:
                            results.append(str(res))
                    except Exception as err:
                        results.append(f"Error on {name}: {str(err)}")
            return " and ".join(results) + "." if results else "Done."
        return msg.get("content", "Understood.")
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 6. AUDIO & SOCKET SERVER
# ==========================================
def transcribe_audio_data(audio_bytes: bytes) -> str:
    mono_16k = convert_to_16k_mono(audio_bytes)
    temp_wav = "/tmp/jarvis_audio.wav"
    with wave.open(temp_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(mono_16k)

    segments, _ = stt_model.transcribe(temp_wav, beam_size=2)
    return " ".join([s.text for s in segments]).strip().lower()

def handle_command(text: str):
    emit_status("LOG", f"Executing: {text}")
    reply = query_llm(text)
    play_single_sentence(reply)

def client_handler(conn):
    global RECORDING_ACTIVE, AUDIO_FRAMES
    with CLIENT_LOCK:
        CONNECTED_CLIENTS.append(conn)
    buffer = ""
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            buffer += data.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                cmd = line.strip()
                if cmd == "SKIP_SPEECH":
                    stop_playback()
                elif cmd == "START_PTT":
                    AUDIO_FRAMES = []
                    RECORDING_ACTIVE = True
                    emit_status("STATE", "RECORDING")
                elif cmd == "STOP_PTT":
                    RECORDING_ACTIVE = False
                    emit_status("STATE", "THINKING")
                    if AUDIO_FRAMES:
                        text = transcribe_audio_data(b"".join(AUDIO_FRAMES))
                        if len(text.strip()) >= 2:
                            handle_command(text)
                    emit_status("STATE", "STANDBY")
                elif cmd:
                    if IS_SPEAKING:
                        stop_playback()
                    handle_command(cmd)
    except Exception:
        pass
    finally:
        with CLIENT_LOCK:
            if conn in CONNECTED_CLIENTS:
                CONNECTED_CLIENTS.remove(conn)
        conn.close()

def socket_server_thread():
    if os.path.exists(SOCKET_PATH):
        try:
            os.remove(SOCKET_PATH)
        except Exception:
            pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)
    os.chmod(SOCKET_PATH, 0o777)
    while True:
        try:
            conn, _ = server.accept()
            threading.Thread(target=client_handler, args=(conn,), daemon=True).start()
        except Exception:
            break

threading.Thread(target=socket_server_thread, daemon=True).start()

def run_jarvis():
    global AUDIO_FRAMES
    with no_alsa_err():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    emit_status("STATE", "STANDBY")
    emit_status("LOG", "Core Online (All Systems Ready)")

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            if RECORDING_ACTIVE:
                AUDIO_FRAMES.append(data)
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        stop_playback()
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    run_jarvis()
