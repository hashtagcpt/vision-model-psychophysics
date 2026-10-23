#!/usr/bin/env python3
"""
run_suite.py - Unified Web Server for Vision Science Psychophysics Suite
Serves the master selection hub and all experiments:
1. Master Intro & Selection Hub (/)
2. Experiment 1: Contrast Sensitivity Function (/vision_experiment_csf/)
3. Experiment 2: Dipper Functions (/vision_experiment_dipper/)
4. Experiment 3: Difference of Gaussians Polarity (/vision_experiment_dog_quest/)
"""

import http.server
import socketserver
import os
import sys

DEFAULT_PORT = 8089

def find_available_port(start_port=8089, max_attempts=20):
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port

def run_server():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    port = find_available_port(DEFAULT_PORT)

    class PsychophysicsHTTPHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Disable caching so changes to JS/CSS are immediately reflected
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

    hub_url = f"http://localhost:{port}/index.html"
    csf_url = f"http://localhost:{port}/vision_experiment_csf/index.html"
    dipper_url = f"http://localhost:{port}/vision_experiment_dipper/index.html"
    dog_url = f"http://localhost:{port}/vision_experiment_dog_quest/index.html"
    model_lab_url = f"http://localhost:{port}/vision_model_psychophysics/index.html"
    viewer_url = f"http://localhost:{port}/results_viewer.html"

    print("=" * 76, flush=True)
    print(" VISION SCIENCE PSYCHOPHYSICS SUITE: UNIFIED RUNNER", flush=True)
    print("=" * 76, flush=True)
    print(f" Serving Directory:  {project_dir}", flush=True)
    print(f" Master Hub URL:     {hub_url}", flush=True)
    print(f" 1. CSF Experiment:  {csf_url}", flush=True)
    print(f" 2. Dipper (TvC):    {dipper_url}", flush=True)
    print(f" 3. DoG Polarity:    {dog_url}", flush=True)
    print(f" 4. Model Lab (AI):  {model_lab_url}", flush=True)
    print(f" 5. Results Viewer:  {viewer_url}", flush=True)
    print("=" * 76, flush=True)
    print(f" Ready! Serving at {hub_url} (Press Ctrl+C to terminate)", flush=True)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), PsychophysicsHTTPHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n Server terminated by user.", flush=True)
            sys.exit(0)

if __name__ == "__main__":
    run_server()
