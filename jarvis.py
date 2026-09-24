import os
import sys
import time
import re
import glob
import shutil
import wave
import signal
import select
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
# 0. SILENCE ALSA / JACK LOG SPAM
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
# 1. HARDWARE & AUDIO CONFIG
# ==========================================
PIPER_MODEL = os.path.expanduser("~/jarvis_os/models/piper/en_US-lessac-medium.onnx")
PIPER_CONFIG = os.path.expanduser("~/jarvis_os/models/piper/en_US-lessac-medium.onnx.json")
WHISPER_SIZE = "base.en"
LLM_MODEL = "qwen2.5:7b"

RATE = 48000
CHANNELS = 2
CHUNK_SIZE = 2048

IS_SPEAKING = False
INTERRUPT_REQUESTED = False
CURRENT_PROC = None
PROC_LOCK = threading.Lock()

PENDING_ACTION = None

print("[Initialization] Loading Whisper speech engine...")
with no_alsa_err():
    stt_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

# ==========================================
# 2. AUDIO DSP & INSTANT KEYBOARD CUTOFF
# ==========================================
def convert_to_16k_mono(raw_bytes: bytes) -> bytes:
    """Downsamples native 48kHz stereo to 16kHz mono int16 cleanly."""
    audio = np.frombuffer(raw_bytes, dtype=np.int16)
    mono = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return mono[::3].tobytes()

def stop_playback():
    """Instantly kills whatever sentence is currently playing."""
    global CURRENT_PROC, INTERRUPT_REQUESTED
    INTERRUPT_REQUESTED = True
    with PROC_LOCK:
        if CURRENT_PROC is not None:
            try:
                os.killpg(os.getpgid(CURRENT_PROC.pid), signal.SIGKILL)
            except Exception:
                pass
            CURRENT_PROC = None

def keyboard_skip_listener():
    """Listens in background for Enter key press on terminal to immediately skip speaking."""
    while True:
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                sys.stdin.readline()
                if IS_SPEAKING:
                    print("\n[User Hit Enter] Skipping speech output...", flush=True)
                    stop_playback()
        except Exception:
            pass

threading.Thread(target=keyboard_skip_listener, daemon=True).start()

def play_single_sentence(sentence: str) -> bool:
    """Plays a single sentence. Returns False if interrupted."""
    global CURRENT_PROC, IS_SPEAKING, INTERRUPT_REQUESTED
    if INTERRUPT_REQUESTED:
        return False

    clean = sentence.replace('"', '\\"').replace("'", "").replace("\n", " ").strip()
    if not clean:
        return True

    IS_SPEAKING = True
    piper_cmd = f'echo "{clean}" | piper --model {PIPER_MODEL} --config {PIPER_CONFIG} --output-raw | aplay -r 22050 -f S16_LE -t raw -q'
    
    proc = subprocess.Popen(piper_cmd, shell=True, preexec_fn=os.setsid)
    with PROC_LOCK:
        CURRENT_PROC = proc

    proc.wait()
    with PROC_LOCK:
        CURRENT_PROC = None

    time.sleep(0.08)
    IS_SPEAKING = False
    return not INTERRUPT_REQUESTED

# ==========================================
# 3. ADVANCED SYSTEM & FILE OPS TOOLS
# ==========================================
def resolve_system_path(raw_path: str) -> Path:
    """Intelligently resolves paths, expanding ~ and matching common home directories."""
    cleaned = raw_path.strip().strip("'\"")
    expanded = Path(os.path.expanduser(cleaned))
    
    if expanded.is_absolute() and expanded.exists():
        return expanded.resolve()

    common_dirs = {
        "downloads": Path.home() / "Downloads",
        "documents": Path.home() / "Documents",
        "desktop": Path.home() / "Desktop",
        "pictures": Path.home() / "Pictures",
        "videos": Path.home() / "Videos",
        "music": Path.home() / "Music",
        "home": Path.home()
    }
    
    low = cleaned.lower().rstrip("/\\")
    if low in common_dirs:
        return common_dirs[low]

    for name, p in common_dirs.items():
        if low.startswith(name + "/"):
            remainder = cleaned[len(name) + 1:]
            return (p / remainder).resolve()

    candidate = Path.home() / cleaned.lstrip("/")
    if candidate.exists():
        return candidate.resolve()

    for p in common_dirs.values():
        sub_candidate = p / cleaned.split("/")[-1]
        if sub_candidate.exists():
            return sub_candidate.resolve()

    return expanded.resolve()

def open_folder(folder_name: str):
    """Opens a system directory or folder in the native file manager (Nautilus)."""
    p = resolve_system_path(folder_name)
    if p.exists() and p.is_dir():
        subprocess.Popen(["xdg-open", str(p)])
        return f"Opened {p.name} folder."
    elif (Path.home() / folder_name.strip()).exists():
        target = Path.home() / folder_name.strip()
        subprocess.Popen(["xdg-open", str(target)])
        return f"Opened {target.name} folder."
    return f"Folder '{folder_name}' not found."

def select_item_in_file_manager(item_name: str):
    """Navigates to and opens a folder or file inside the currently active Nautilus/Files window using type-ahead search."""
    if not shutil.which("xdotool"):
        return "xdotool is not installed. Please run: sudo apt install xdotool"

    clean_item = item_name.strip().strip("'\"")
    clean_item = re.sub(r'\s+(folder|directory|file)$', '', clean_item, flags=re.I).strip()

    script = f"""
    xdotool key --clearmodifiers Escape
    sleep 0.15
    xdotool type --delay 40 "{clean_item}"
    sleep 0.35
    xdotool key Return
    """
    subprocess.run(script, shell=True)
    return f"Opened '{clean_item}' in the active window."

def move_file_or_folder(source: str, destination: str, filename: str = None):
    """Moves files or directories reliably across user directories."""
    try:
        src_path = resolve_system_path(source)
        dest_path = resolve_system_path(destination)

        if filename:
            target_file = src_path / filename if src_path.is_dir() else src_path
            if not target_file.exists():
                for sub in [Path.home() / "Downloads", Path.home() / "Documents", Path.home() / "Desktop"]:
                    if (sub / filename).exists():
                        target_file = sub / filename
                        break
            src_path = target_file

        if not src_path.exists():
            return f"Source file or folder '{source}' not found."

        dest_path.mkdir(parents=True, exist_ok=True)
        final_dest = dest_path / src_path.name if dest_path.is_dir() else dest_path

        shutil.move(str(src_path), str(final_dest))
        return f"Moved {src_path.name} to {dest_path.name}."
    except Exception as e:
        return f"Failed to move: {str(e)}"

def copy_file_or_folder(source: str, destination: str, filename: str = None):
    """Copies files or directories reliably across user directories."""
    try:
        src_path = resolve_system_path(source)
        dest_path = resolve_system_path(destination)

        if filename:
            target_file = src_path / filename if src_path.is_dir() else src_path
            if not target_file.exists():
                for sub in [Path.home() / "Downloads", Path.home() / "Documents", Path.home() / "Desktop"]:
                    if (sub / filename).exists():
                        target_file = sub / filename
                        break
            src_path = target_file

        if not src_path.exists():
            return f"Source file or folder '{source}' not found."

        dest_path.mkdir(parents=True, exist_ok=True)
        final_dest = dest_path / src_path.name if dest_path.is_dir() else dest_path

        if src_path.is_dir():
            shutil.copytree(str(src_path), str(final_dest), dirs_exist_ok=True)
        else:
            shutil.copy2(str(src_path), str(final_dest))

        return f"Copied {src_path.name} to {dest_path.name}."
    except Exception as e:
        return f"Failed to copy: {str(e)}"

def delete_file_or_folder(target: str):
    """Safely removes or trashes a file, folder, wildcard path, or entire directory contents."""
    try:
        raw = target.strip().strip("'\"")

        # 1. Handle wildcard patterns (e.g. ~/Downloads/* or Downloads/*)
        if "*" in raw:
            base_dir_str = raw.split("*")[0].rstrip("/\\")
            base_dir = resolve_system_path(base_dir_str)
            if not base_dir.exists() or not base_dir.is_dir():
                return f"Directory '{base_dir_str}' not found."

            protected = [Path("/"), Path.home(), Path("/etc"), Path("/usr"), Path("/boot"), Path("/bin"), Path("/var")]
            if base_dir in protected:
                return "Action blocked: Deleting root or home contents directly is protected."

            items = [p for p in base_dir.iterdir() if not p.name.startswith(".")]
            if not items:
                return f"{base_dir.name} is already empty."

            deleted_count = 0
            for item in items:
                if shutil.which("gio"):
                    subprocess.run(["gio", "trash", str(item)], capture_output=True)
                else:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                deleted_count += 1
            return f"Deleted {deleted_count} items in {base_dir.name}."

        # 2. Check if user passed a directory and wants to delete its contents
        target_path = resolve_system_path(raw)

        protected = [Path("/"), Path.home(), Path("/etc"), Path("/usr"), Path("/boot"), Path("/bin"), Path("/var")]
        if target_path in protected:
            return f"Action blocked: '{raw}' is protected."

        if not target_path.exists():
            # Check if user meant folder contents by passing folder name
            for folder_name in ["downloads", "documents", "desktop", "pictures", "videos", "music"]:
                if folder_name in raw.lower():
                    parent = resolve_system_path(folder_name)
                    items = [p for p in parent.iterdir() if not p.name.startswith(".")]
                    if not items:
                        return f"{parent.name} is already empty."
                    deleted_count = 0
                    for item in items:
                        if shutil.which("gio"):
                            subprocess.run(["gio", "trash", str(item)], capture_output=True)
                        else:
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()
                        deleted_count += 1
                    return f"Deleted {deleted_count} items in {parent.name}."
            return f"File or folder '{raw}' not found."

        # 3. Single target deletion
        if shutil.which("gio"):
            res = subprocess.run(["gio", "trash", str(target_path)], capture_output=True)
            if res.returncode == 0:
                return f"Moved {target_path.name} to system trash."

        if target_path.is_file() or target_path.is_symlink():
            target_path.unlink()
        elif target_path.is_dir():
            shutil.rmtree(target_path)

        return f"Deleted {target_path.name}."
    except Exception as e:
        return f"Failed to delete: {str(e)}"

def get_current_time_and_date():
    """Reads the exact real-time clock and calendar date directly from Linux system."""
    now = datetime.now()
    time_str = now.strftime("%I:%M %p").lstrip('0')
    date_str = now.strftime("%A, %B %d, %Y")
    return f"The current time is {time_str}, and today is {date_str}."

def format_file_size(size_in_bytes: int) -> str:
    """Converts bytes to a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}" if unit != 'B' else f"{int(size_in_bytes)} B"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} PB"

def get_item_disk_size(path: Path) -> int:
    """Calculates file size or directory size."""
    try:
        if path.is_file() or path.is_symlink():
            return path.stat().st_size
        elif path.is_dir():
            total = 0
            for entry in path.rglob('*'):
                try:
                    if entry.is_file():
                        total += entry.stat().st_size
                except (PermissionError, FileNotFoundError):
                    continue
            return total
    except (PermissionError, FileNotFoundError):
        return 0
    return 0

def list_directory_files(directory: str = "~", sorted_by: str = None, order: str = "desc", **kwargs):
    """Lists files and folders, supporting sorting by space/size or date with human-readable space calculation."""
    target_dir = resolve_system_path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        return f"Directory {directory} does not exist."

    sort_key = (sorted_by or kwargs.get("sort_by") or kwargs.get("by") or "").lower().strip()

    try:
        entries = [item for item in target_dir.iterdir() if not item.name.startswith(".")]
        if not entries:
            return f"The directory {target_dir.name} is empty."

        if any(k in sort_key for k in ["size", "space", "storage", "bytes", "large", "heavy"]):
            items_with_size = [(item, get_item_disk_size(item)) for item in entries]
            reverse_order = False if "asc" in order.lower() else True
            items_with_size.sort(key=lambda x: x[1], reverse=reverse_order)
            
            top_items = items_with_size[:6]
            formatted_list = [f"{item.name} ({format_file_size(sz)})" for item, sz in top_items]
            result_str = ", ".join(formatted_list)
            
            if len(items_with_size) > 6:
                return f"Files by space: {result_str}, and {len(items_with_size) - 6} more."
            return f"Files by space: {result_str}."

        elif any(k in sort_key for k in ["date", "time", "recent", "modified"]):
            reverse_order = False if "asc" in order.lower() else True
            entries.sort(key=lambda x: x.stat().st_mtime, reverse=reverse_order)
            names = [item.name for item in entries[:8]]
            return "Files: " + ", ".join(names)

        else:
            entries.sort(key=lambda x: x.name.lower())
            names = [item.name for item in entries[:8]]
            if len(entries) > 8:
                return f"Directory has {len(entries)} items: " + ", ".join(names) + ", and more."
            return "Files: " + ", ".join(names)

    except Exception as e:
        return f"Could not list directory: {str(e)}"

def run_terminal_command(command: str):
    """Executes bash commands directly and returns concise terminal output."""
    blacklist = ["rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:"]
    if any(b in command for b in blacklist):
        return "Command blocked by security guardrails."
    
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=8)
        out = (res.stdout or res.stderr).strip()
        if not out:
            return "Command executed with no output."
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        if len(lines) > 6:
            summary = ", ".join(lines[:6])
            return f"Found {len(lines)} items: {summary}, and more."
        return ", ".join(lines)
    except subprocess.TimeoutExpired:
        return "Command timed out after 8 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

def type_query_into_site(target_site: str, text_to_type: str, press_enter: bool = True):
    """Opens a website or web app and types the query into its input box."""
    if not shutil.which("xdotool"):
        return "xdotool is not installed. Please run: sudo apt install xdotool"

    site_clean = target_site.lower().strip()
    site_map = {
        "gemini": "https://gemini.google.com/app",
        "chatgpt": "https://chatgpt.com",
        "claude": "https://claude.ai",
        "youtube": "https://youtube.com",
        "google": "https://google.com",
        "reddit": "https://reddit.com",
        "github": "https://github.com",
        "twitter": "https://x.com",
        "x": "https://x.com"
    }

    url = site_map.get(site_clean, None)
    if not url:
        if site_clean.startswith("http://") or site_clean.startswith("https://"):
            url = site_clean
        else:
            url = f"https://{site_clean}.com"

    subprocess.Popen(["xdg-open", url])

    def _type_worker():
        time.sleep(2.6)
        subprocess.run("xdotool click 1", shell=True)
        time.sleep(0.1)
        subprocess.run("xdotool key --clearmodifiers Tab", shell=True)
        time.sleep(0.1)

        clean_text = text_to_type.replace('"', '\\"').strip()
        enter_cmd = "\nsleep 0.2\nxdotool key Return" if press_enter else ""
        script = f"""
        xdotool type --delay 25 "{clean_text}"{enter_cmd}
        """
        subprocess.run(script, shell=True)

    threading.Thread(target=_type_worker, daemon=True).start()
    return f"Opening {site_clean.title()} and entering your prompt: '{text_to_type}'."

def browser_action(action: str):
    """Controls browser navigation."""
    if not shutil.which("xdotool"):
        return "xdotool is not installed. Please run: sudo apt install xdotool"

    act = action.lower().strip()
    actions = {
        "close_tab": "ctrl+w",
        "close_window": "ctrl+shift+w",
        "back": "alt+Left",
        "forward": "alt+Right",
        "refresh": "ctrl+r",
        "new_tab": "ctrl+t",
        "scroll_down": "Page_Down",
        "scroll_up": "Page_Up",
        "fullscreen": "f",
        "toggle_media": "space"
    }

    if act not in actions:
        return f"Unknown browser action: {action}"

    key = actions[act]
    subprocess.run(f"xdotool key --clearmodifiers {key}", shell=True)
    return f"Browser: {act.replace('_', ' ')} executed."

def browser_click_link(target: str):
    """Clicks a link by position or visible text."""
    if not shutil.which("xdotool"):
        return "xdotool is not installed. Run 'sudo apt install xdotool' to enable clicking."

    target_clean = target.lower().strip()
    ordinal_map = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
        "fifth": 5, "5th": 5
    }

    pos = None
    for word, num in ordinal_map.items():
        if word in target_clean:
            pos = num
            break

    if pos is not None:
        tab_presses = " ".join(["key Tab"] * (pos + 2))
        script = f"""
        xdotool key --clearmodifiers Escape
        sleep 0.1
        xdotool {tab_presses}
        sleep 0.1
        xdotool key Return
        """
        subprocess.run(script, shell=True)
        return f"Clicked the {target_clean}."

    clean_kw = target.replace('"', '\\"').strip()
    script = f"""
    xdotool key --clearmodifiers ctrl+f
    sleep 0.1
    xdotool type --delay 15 "{clean_kw}"
    sleep 0.1
    xdotool key Return
    sleep 0.1
    xdotool key Escape
    sleep 0.1
    xdotool key Return
    """
    subprocess.run(script, shell=True)
    return f"Clicked '{target}'."

def start_gmail_login():
    """Opens Gmail login page and begins guided sign-in."""
    global PENDING_ACTION
    subprocess.Popen(["xdg-open", "https://accounts.google.com/signin/v2/identifier?service=mail"])
    PENDING_ACTION = {"state": "waiting_email"}
    return "I've opened the Gmail login page. What is your email address or account name?"

def enter_browser_text(text: str, press_enter: bool = True):
    """Types text directly into the focused field in the browser."""
    if not shutil.which("xdotool"):
        return "xdotool is not installed."
    clean_txt = text.replace('"', '\\"').strip()
    enter_cmd = "\nsleep 0.2\nxdotool key Return" if press_enter else ""
    script = f"""
    sleep 0.4
    xdotool type --delay 45 "{clean_txt}"{enter_cmd}
    """
    subprocess.run(script, shell=True)
    return f"Typed '{text}'."

def web_search(query: str, platform: str = "auto"):
    """Searches YouTube or Google directly, opening in a new tab/window."""
    q_clean = query.lower()
    if "youtube" in platform.lower() or "youtube" in q_clean or "video" in q_clean or "song" in q_clean:
        cleaned_query = re.sub(r'\b(on youtube|in youtube|youtube|videos on|video of|search for|play)\b', '', query, flags=re.I).strip()
        final_query = cleaned_query if cleaned_query else query
        encoded = urllib.parse.quote_plus(final_query)
        url = f"https://www.youtube.com/results?search_query={encoded}"
        subprocess.Popen(["xdg-open", url])
        return f"Searching YouTube for '{final_query}'"

    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.google.com/search?q={encoded}"
    subprocess.Popen(["xdg-open", url])
    return f"Searching Google for '{query}'"

def launch_app(app_name: str):
    """Launches desktop applications, user folders, or websites."""
    clean = app_name.lower().strip()

    common_folders = ["documents", "downloads", "desktop", "pictures", "videos", "music", "home"]
    if clean in common_folders or any(clean == f"{cf} folder" for cf in common_folders):
        base_folder = clean.replace(" folder", "").strip()
        return open_folder(base_folder)

    web_targets = {
        "google": "https://google.com",
        "youtube": "https://youtube.com",
        "gemini": "https://gemini.google.com",
        "chatgpt": "https://chatgpt.com",
        "github": "https://github.com",
        "reddit": "https://reddit.com"
    }
    if clean in web_targets:
        subprocess.Popen(["xdg-open", web_targets[clean]])
        return f"Opened {clean.title()}."

    app_map = {
        "app store": ["snap-store", "gnome-software", "ubuntu-software"],
        "store": ["snap-store", "gnome-software"],
        "software": ["snap-store", "gnome-software"],
        "files": ["nautilus"],
        "file manager": ["nautilus"],
        "text editor": ["gnome-text-editor", "gedit", "kate"],
        "editor": ["gnome-text-editor", "gedit"],
        "terminal": ["gnome-terminal", "ptyxis", "alacritty", "kitty"],
        "calculator": ["gnome-calculator"],
        "settings": ["gnome-control-center"],
        "browser": ["google-chrome", "firefox", "brave-browser", "chromium"],
        "chrome": ["google-chrome"],
        "code": ["code"]
    }
    candidates = app_map.get(clean, [clean])

    for c in candidates:
        if subprocess.run(["gtk-launch", c], capture_output=True).returncode == 0:
            return f"Opened {app_name}."
    for c in candidates:
        if shutil.which(c):
            subprocess.Popen([c], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Opened {app_name}."

    desktop_dirs = [Path("/usr/share/applications"), Path.home() / ".local/share/applications"]
    for d in desktop_dirs:
        if d.exists():
            for f in d.glob("*.desktop"):
                if clean in f.name.lower():
                    subprocess.Popen(["gtk-launch", f.stem], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return f"Opened {app_name}."
    return f"Could not find application '{app_name}'."

def close_application(app_name: str):
    """Closes applications cleanly without permission errors."""
    app_clean = app_name.lower().strip()
    alias_map = {
        "files": ["nautilus", "org.gnome.Nautilus"],
        "file manager": ["nautilus"],
        "text editor": ["gnome-text-editor", "gedit"],
        "editor": ["gnome-text-editor", "gedit"],
        "browser": ["chrome", "firefox", "brave"],
        "app store": ["snap-store", "gnome-software"],
        "store": ["snap-store", "gnome-software"],
        "code": ["code"],
        "terminal": ["gnome-terminal-server", "ptyxis"]
    }
    targets = alias_map.get(app_clean, [app_clean])

    gnome_bus_map = {
        "files": "org.gnome.Nautilus",
        "text editor": "org.gnome.TextEditor",
        "app store": "org.gnome.Software"
    }
    if app_clean in gnome_bus_map:
        bus = gnome_bus_map[app_clean]
        res = subprocess.run([
            "gdbus", "call", "--session", "--dest", bus,
            "--object-path", f"/{bus.replace('.', '/')}",
            "--method", "org.gtk.Actions.Activate", "quit", "[]", "{}"
        ], capture_output=True)
        if res.returncode == 0:
            return f"Closed {app_name}."

    current_user = os.environ.get("USER", "")
    for t in targets:
        pkill_cmd = ["pkill", "-15", "-u", current_user, "-f", t] if current_user else ["pkill", "-15", "-f", t]
        res = subprocess.run(pkill_cmd, capture_output=True)
        if res.returncode == 0:
            return f"Closed {app_name}."
    return f"No running window found for {app_name}."

def open_local_file(filepath: str, app: str = None):
    p = resolve_system_path(filepath)
    if not p.exists():
        return f"File {filepath} not found."
    if p.is_dir():
        return open_folder(str(p))
    if (app and "code" in app.lower()) or (p.suffix in [".py", ".js", ".ts", ".json"] and shutil.which("code")):
        subprocess.Popen(["code", str(p)])
        return f"Opened {p.name} in VS Code."
    subprocess.Popen(["xdg-open", str(p)])
    return f"Opened {p.name}."

def set_system_volume(percent: int = None, delta: int = None):
    subprocess.run("wpctl set-mute @DEFAULT_AUDIO_SINK@ 0 2>/dev/null || pactl set-sink-mute @DEFAULT_SINK@ 0 2>/dev/null", shell=True)
    if delta is not None:
        sign = "+" if delta > 0 else "-"
        subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{abs(delta)}%{sign}"], capture_output=True)
        return f"Volume adjusted by {delta} percent."
    if percent is not None:
        percent = max(0, min(100, int(percent)))
        subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", str(round(percent/100.0, 2))], capture_output=True)
        return f"Volume set to {percent} percent."
    return "Could not determine volume level."

def get_storage_status():
    total, used, free = shutil.disk_usage("/")
    return f"You have {free // (2**30)} GB free out of {total // (2**30)} GB."

def get_battery_status():
    for b in Path("/sys/class/power_supply").glob("BAT*"):
        cap = (b / "capacity").read_text().strip() if (b / "capacity").exists() else None
        stat = (b / "status").read_text().strip() if (b / "status").exists() else None
        if cap:
            return f"Battery is at {cap}% and {stat}."
    return "Battery info unavailable."

AVAILABLE_TOOLS = {
    "open_folder": open_folder,
    "select_item_in_file_manager": select_item_in_file_manager,
    "move_file_or_folder": move_file_or_folder,
    "copy_file_or_folder": copy_file_or_folder,
    "delete_file_or_folder": delete_file_or_folder,
    "get_current_time_and_date": get_current_time_and_date,
    "list_directory_files": list_directory_files,
    "run_terminal_command": run_terminal_command,
    "type_query_into_site": type_query_into_site,
    "enter_browser_text": enter_browser_text,
    "browser_action": browser_action,
    "browser_click_link": browser_click_link,
    "start_gmail_login": start_gmail_login,
    "web_search": web_search,
    "launch_app": launch_app,
    "close_application": close_application,
    "open_local_file": open_local_file,
    "set_system_volume": set_system_volume,
    "get_storage_status": get_storage_status,
    "get_battery_status": get_battery_status
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "delete_file_or_folder",
            "description": "Deletes or moves a file, folder, wildcard path (e.g. '~/Downloads/*'), or entire directory contents to the trash safely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Path, file, or folder pattern to delete (e.g. '~/Downloads/*', 'Downloads', 'test.txt')"
                    }
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_folder",
            "description": "Opens a system folder or directory in the file manager (e.g. 'documents', 'downloads', 'desktop', 'pictures', 'videos', 'music').",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_name": {"type": "string", "description": "Name or path of folder, e.g. 'documents', 'downloads', 'music'"}
                },
                "required": ["folder_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_item_in_file_manager",
            "description": "Selects, clicks, or opens a folder or file by name inside the currently open Files / Nautilus window (e.g. 'click on snap folder', 'open snap folder', 'open test.py').",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string", "description": "Name of the folder or file to click, e.g. 'snap', 'projects', 'notes'"}
                },
                "required": ["item_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_file_or_folder",
            "description": "Moves a file or folder from a source folder/path to a destination folder/path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source path or folder, e.g. 'Downloads/pass.txt' or 'Downloads'"},
                    "destination": {"type": "string", "description": "Destination directory, e.g. 'Documents' or '~/Desktop'"},
                    "filename": {"type": "string", "description": "Optional name of file if source is only a folder, e.g. 'pass.txt'"}
                },
                "required": ["source", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "copy_file_or_folder",
            "description": "Copies a file or folder from a source folder/path to a destination folder/path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source file or folder path, e.g. 'Downloads/report.pdf'"},
                    "destination": {"type": "string", "description": "Destination folder, e.g. 'Documents'"},
                    "filename": {"type": "string", "description": "Optional file name, e.g. 'report.pdf'"}
                },
                "required": ["source", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time_and_date",
            "description": "Reads the exact real-time clock and current date from the operating system.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory_files",
            "description": "Lists files/folders in a directory. Supports sorting by space/size or date, and reports space taken up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Target folder, e.g. '~' or '~/Downloads'"},
                    "sorted_by": {
                        "type": "string",
                        "enum": ["space", "size", "date", "name"],
                        "description": "Property to sort by, like 'space' or 'size'"
                    },
                    "order": {
                        "type": "string",
                        "enum": ["desc", "asc"],
                        "description": "Sorting direction (desc = largest/newest first)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal_command",
            "description": "Executes shell commands (e.g. 'ls', 'pwd', 'df', 'free') and returns output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The exact shell command to run"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type_query_into_site",
            "description": "Opens a web app or platform (like Gemini, ChatGPT, Claude, YouTube, etc.) and types a specified prompt into its input box.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_site": {
                        "type": "string",
                        "description": "Target site name, e.g. 'gemini', 'chatgpt', 'youtube'"
                    },
                    "text_to_type": {
                        "type": "string",
                        "description": "Prompt or message to type"
                    },
                    "press_enter": {
                        "type": "boolean",
                        "description": "Whether to hit Enter after typing"
                    }
                },
                "required": ["target_site", "text_to_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "enter_browser_text",
            "description": "Types text directly into the focused field in the browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to type"},
                    "press_enter": {"type": "boolean"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_action",
            "description": "Controls browser: close tab, close window, go back, forward, refresh, scroll, play/pause video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "close_tab",
                            "close_window",
                            "back",
                            "forward",
                            "refresh",
                            "new_tab",
                            "scroll_down",
                            "scroll_up",
                            "fullscreen",
                            "toggle_media"
                        ]
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click_link",
            "description": "Clicks a link, button, or search result by position ('first link', 'second video') or visible text label in the browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "e.g. 'first link', 'second video', 'Shorts'"}
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_gmail_login",
            "description": "Opens Gmail login page and begins guided sign-in.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches YouTube or Google. Set platform='youtube' when query relates to videos, music, or songs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query keywords"},
                    "platform": {"type": "string", "enum": ["youtube", "google", "auto"]}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "launch_app",
            "description": "Launches applications (e.g. files, text editor, terminal, chrome).",
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
            "name": "close_application",
            "description": "Closes an entire desktop application or process window cleanly.",
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
            "name": "set_system_volume",
            "description": "Adjusts or sets master volume level.",
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
            "name": "open_local_file",
            "description": "Opens a document or code file in editor.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_storage_status",
            "description": "Checks free hard disk drive space.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_battery_status",
            "description": "Reads laptop battery level.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# ==========================================
# 4. LLM INFERENCE (MULTI-TOOL ENABLED)
# ==========================================
SYSTEM_PROMPT = (
    "You are Jarvis, an autonomous desktop assistant running natively on Linux. "
    "Rules:\n"
    "1. Respond in strictly 1 short sentence.\n"
    "2. NEVER output raw Python function calls or text like 'delete_file_or_folder(...)' into content. "
    "You MUST invoke actions via the provided tool_calls mechanism.\n"
    "3. FILE MANAGEMENT:\n"
    "   - When asked to delete all files in a folder (e.g. 'delete all files in downloads'), "
    "call 'delete_file_or_folder' with target='~/Downloads/*'.\n"
    "   - To delete a single file/folder, call 'delete_file_or_folder' with the target path.\n"
    "   - To move a file/folder, call 'move_file_or_folder' with source and destination.\n"
    "   - To copy a file/folder, call 'copy_file_or_folder' with source and destination.\n"
    "4. DIRECTORIES & FOLDERS:\n"
    "   - When asked to open a folder like 'open documents', 'open downloads', call 'open_folder'.\n"
    "   - When asked to click or select an item inside the file manager, call 'select_item_in_file_manager'.\n"
    "5. MULTI-COMMAND EXECUTION: When the user asks to perform multiple tasks in one sentence, "
    "call EVERY necessary tool in your tool_calls response list.\n"
    "6. For asking about current time or date, call 'get_current_time_and_date'.\n"
    "7. If the user asks to list files, call list_directory_files.\n"
    "8. If the user asks to execute/run a bash command, call run_terminal_command.\n"
    "9. If the user asks to type into a website, call type_query_into_site.\n"
    "10. For closing a tab, call browser_action with action='close_tab'. For going back, action='back'.\n"
    "11. For playing/pausing media, call browser_action with action='toggle_media'.\n"
    "12. For general apps (browser, terminal, text editor), call launch_app; to terminate whole apps, call close_application."
)

def query_llm(prompt: str) -> str:
    print(f"\n[Interpreting Intent]: {prompt}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    try:
        response = ollama.chat(model=LLM_MODEL, messages=messages, tools=TOOL_SCHEMAS)
        msg = response.get("message", {})
        
        tool_calls = msg.get("tool_calls", [])
        
        # Fallback regex parser if the local model outputs text instead of structured tool_calls
        content = msg.get("content", "").strip()
        if not tool_calls and content:
            m = re.match(r'(\w+)\((.*)\)', content, re.DOTALL)
            if m:
                func_name = m.group(1)
                args_str = m.group(2)
                if func_name in AVAILABLE_TOOLS:
                    parsed_args = {}
                    for match in re.finditer(r'(\w+)=["\']([^"\']*)["\']', args_str):
                        parsed_args[match.group(1)] = match.group(2)
                    tool_calls = [{
                        "function": {
                            "name": func_name,
                            "arguments": parsed_args
                        }
                    }]

        if tool_calls:
            results = []
            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                print(f"[Executing Tool Call]: {name}({args})")
                if name in AVAILABLE_TOOLS:
                    try:
                        res = AVAILABLE_TOOLS[name](**args)
                        if res:
                            results.append(str(res))
                    except Exception as err:
                        results.append(f"Error on {name}: {str(err)}")
            if results:
                return " and ".join(results) + "."
            return "Executed requested actions."
            
        return content if content else "Understood."
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 5. AUDIO TRANSCRIPTION & CAPTURE
# ==========================================
def transcribe_audio_data(audio_bytes: bytes) -> str:
    mono_16k = convert_to_16k_mono(audio_bytes)
    temp_wav = "/tmp/jarvis_audio.wav"
    with wave.open(temp_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(mono_16k)
        
    segments, _ = stt_model.transcribe(
        temp_wav,
        beam_size=2,
        initial_prompt="Hey Jarvis, delete all files in downloads, move pass.txt, open documents, click snap, copy, delete."
    )
    return " ".join([s.text for s in segments]).strip().lower()

def check_for_barge_in(stream, trigger_thresh: int, silence_thresh: int) -> str:
    """Listens during inter-sentence pause for spoken interruptions."""
    check_chunks = int((RATE / CHUNK_SIZE) * 0.35)
    for _ in range(check_chunks):
        if INTERRUPT_REQUESTED:
            return ""
        chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        energy = int(np.abs(np.frombuffer(chunk, dtype=np.int16)).mean())
        if energy > trigger_thresh:
            print(f"\n[Interruption Detected - Energy: {energy}] Listening to new command...", flush=True)
            recorded = [chunk]
            silent_count = 0
            silence_limit = int((RATE / CHUNK_SIZE) * 0.8)
            max_rec = int((RATE / CHUNK_SIZE) * 5)
            while len(recorded) < max_rec:
                if INTERRUPT_REQUESTED:
                    return ""
                c = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                recorded.append(c)
                if int(np.abs(np.frombuffer(c, dtype=np.int16)).mean()) < silence_thresh:
                    silent_count += 1
                else:
                    silent_count = 0
                if len(recorded) > int((RATE / CHUNK_SIZE) * 0.6) and silent_count > silence_limit:
                    break
            return transcribe_audio_data(b"".join(recorded))
    return ""

def speak_with_interruption(text: str, stream, trigger_thresh: int, silence_thresh: int):
    """Streams sentences sequentially and listens for Enter key or spoken interruptions."""
    global INTERRUPT_REQUESTED
    INTERRUPT_REQUESTED = False

    print(f"\n[Jarvis Voice]: {text}")
    print("[Tip]: Press [Enter] anytime to immediately skip speaking.")
    sentences = re.split(r'(?<=[.!?]) +', text)

    for i, s in enumerate(sentences):
        if INTERRUPT_REQUESTED:
            break

        play_single_sentence(s)

        if i < len(sentences) - 1:
            heard = check_for_barge_in(stream, trigger_thresh, silence_thresh)
            if heard:
                print(f"[Interrupted With]: \"{heard}\"")
                stop_playback()
                stop_words = ["stop", "shut up", "okay", "got it", "quiet", "cancel", "thanks"]
                if not any(w in heard for w in stop_words):
                    handle_command(heard, stream, trigger_thresh, silence_thresh)
                return

# ==========================================
# 6. MAIN ENGINE & CONTEXT DISPATCHER
# ==========================================
def handle_command(text: str, stream, trigger_thresh: int, silence_thresh: int):
    global PENDING_ACTION

    if PENDING_ACTION:
        state = PENDING_ACTION.get("state")
        
        if state == "waiting_email":
            email_val = text.replace(" at ", "@").replace(" dot ", ".").replace(" ", "").strip()
            print(f"[Gmail Flow] Entering Account: {email_val}")
            enter_browser_text(email_val, press_enter=True)
            PENDING_ACTION = {"state": "waiting_password"}
            speak_with_interruption("Account entered. What is your password?", stream, trigger_thresh, silence_thresh)
            return

        elif state == "waiting_password":
            pwd_val = text.replace(" ", "").strip()
            print("[Gmail Flow] Entering Password...")
            enter_browser_text(pwd_val, press_enter=True)
            PENDING_ACTION = None
            speak_with_interruption("Password entered. You are logging in.", stream, trigger_thresh, silence_thresh)
            return

    wake_words = ["hey jarvis", "jarvis", "jazza", "jollis", "travis"]
    action_starters = [
        "delete", "remove", "trash", "clear", "clean",
        "documents", "downloads", "pictures", "desktop", "music", "videos",
        "move", "copy",
        "time", "date", "day", "today", "clock",
        "execute", "run", "list", "ls", "dir", "show",
        "open", "close", "click", "set", "turn", "search",
        "what", "how", "storage", "battery", "tell", "explain", "who", "why",
        "go", "play", "pause", "resume", "login", "log in", "scroll", "refresh", "reload", "type"
    ]

    has_wake = any(w in text for w in wake_words)
    has_action = any(text.startswith(act) for act in action_starters)

    if has_wake or has_action:
        cmd = text
        for w in wake_words:
            cmd = cmd.replace(w, "").strip()
        if len(cmd) > 1:
            reply = query_llm(cmd)
            speak_with_interruption(reply, stream, trigger_thresh, silence_thresh)

def run_jarvis():
    with no_alsa_err():
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

    print("[Calibration] Calibrating microphone (remain quiet)...")
    energies = []
    for _ in range(int((RATE / CHUNK_SIZE) * 1.5)):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        energies.append(np.abs(np.frombuffer(data, dtype=np.int16)).mean())
    
    ambient = int(np.mean(energies))
    trigger_thresh = ambient + 3500
    silence_thresh = ambient + 1200
    print(f"[Ready] Ambient: {ambient} | Trigger: {trigger_thresh} | Silence: {silence_thresh}")

    play_single_sentence("Jarvis is online and ready.")
    print("\n[System]: Listening for voice input...")

    buffer = []
    buffer_len = int((RATE / CHUNK_SIZE) * 1.2)
    silence_limit = int((RATE / CHUNK_SIZE) * 0.9)

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            if IS_SPEAKING:
                buffer.clear()
                continue

            energy = int(np.abs(np.frombuffer(data, dtype=np.int16)).mean())
            meter = "#" * min(int(energy / 500), 25)
            print(f"\rLevel: {energy:5d} [{meter:<25}]", end="", flush=True)

            if energy > trigger_thresh:
                print(f"\n[Voice Detected - Energy: {energy}] Recording...", flush=True)
                frames = []
                silent_count = 0
                max_frames = int((RATE / CHUNK_SIZE) * 6)

                while len(frames) < max_frames:
                    chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(chunk)
                    chunk_energy = np.abs(np.frombuffer(chunk, dtype=np.int16)).mean()
                    if chunk_energy < silence_thresh:
                        silent_count += 1
                    else:
                        silent_count = 0
                    if len(frames) > int((RATE / CHUNK_SIZE) * 0.8) and silent_count > silence_limit:
                        break

                full_audio = b"".join(buffer) + b"".join(frames)
                buffer.clear()

                text = transcribe_audio_data(full_audio)
                print(f"[Heard]: \"{text}\"")

                if len(text.strip()) >= 2:
                    handle_command(text, stream, trigger_thresh, silence_thresh)

                print("\n[System]: Listening...")

            else:
                buffer.append(data)
                if len(buffer) > buffer_len:
                    buffer.pop(0)

    except KeyboardInterrupt:
        stop_playback()
        print("\nShutting down Jarvis...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    run_jarvis()
