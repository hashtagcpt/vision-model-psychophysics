"""
run_experiment.py - Local Web Server & Runner for Vision Science Experiment
Launches the PsychoJS DoG QUEST+ experiment in your default browser.
"""

import sys
import os
import webbrowser
import http.server
import socketserver
from pathlib import Path

DIR = Path(__file__).resolve().parent
PORT = 8088

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIR), **kwargs)

def run():
    url = f"http://127.0.0.1:{PORT}/index.html"
    print("=================================================================")
    print(" PsychoJS Vision Science Experiment Runner")
    print(" Difference of Gaussians (DoG) with Multidimensional QUEST+")
    print("=================================================================")
    print(f"[*] Serving files from : {DIR}")
    print(f"[*] Local URL          : {url}")
    print(f"[*] Opening in your default browser...")
    print("-----------------------------------------------------------------")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("=================================================================\n")

    # Open browser automatically
    webbrowser.open(url)

    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Server stopped.")

if __name__ == "__main__":
    run()
