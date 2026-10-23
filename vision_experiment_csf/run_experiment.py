#!/usr/bin/env python3
"""
run_experiment.py - Local Web Server for Vision Science CSF Experiment
Serves the experiment interface and automatically opens the default web browser.
"""

import http.server
import socketserver
import webbrowser
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

    url = f"http://localhost:{port}/index.html"

    print("=" * 72, flush=True)
    print(" VISION SCIENCE EXPERIMENT: Contrast Sensitivity Function (CSF)", flush=True)
    print("=" * 72, flush=True)
    print(f" Serving directory: {project_dir}", flush=True)
    print(f" Local URL:         {url}", flush=True)
    print(f" Display Bit-Depth: 8-Bit / 10-Bit with High-Frequency Spatial Dithering", flush=True)
    print(f" Stimulus Type:     Constant-Envelope Gabor (sigma = 0.5 deg)", flush=True)
    print(f" Task Paradigm:     Two-Interval Forced Choice (2IFC 2AFC) Detection", flush=True)
    print(f" Spatial Frequencies: 0.125 to 32 cpd (8 Full Octaves)", flush=True)
    print(f" Adaptive Method:   Watson (2017) QUEST+ Bayesian Staircase", flush=True)
    print(f" Data Export:       Consolidated Full CSV with Complete Hardware Profile", flush=True)
    print("=" * 72, flush=True)
    print(f" Ready! Serving at {url} (Press Ctrl+C to terminate)", flush=True)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), PsychophysicsHTTPHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n Server terminated by user.", flush=True)
            sys.exit(0)

if __name__ == "__main__":
    run_server()
