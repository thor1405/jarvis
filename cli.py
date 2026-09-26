#!/usr/bin/env python3
import os
import sys

PIPE = "/tmp/jarvis.pipe"

if not os.path.exists(PIPE):
    try:
        os.mkfifo(PIPE)
    except Exception:
        pass

print("=== Jarvis Text Control Terminal ===")
print("Type any command and hit Enter. Press Ctrl+C or type 'exit' to quit.\n")

try:
    while True:
        cmd = input("Jarvis > ").strip()
        if not cmd:
            continue
        if cmd.lower() in ["exit", "quit", "q"]:
            break
        try:
            with open(PIPE, "w", encoding="utf-8") as fifo:
                fifo.write(cmd + "\n")
        except Exception as e:
            print(f"[Error writing to pipe]: {e}")
except (KeyboardInterrupt, EOFError):
    print("\nExiting CLI...")
