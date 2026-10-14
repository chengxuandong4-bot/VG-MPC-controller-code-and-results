#!/usr/bin/env python3
"""Calls llm_generate_v2.py in --fullconfig mode."""
import sys, os, subprocess
here = os.path.dirname(os.path.abspath(__file__))
cmd = [sys.executable, os.path.join(here, 'llm_generate_v2.py')] + sys.argv[1:] + ['--fullconfig']
sys.exit(subprocess.run(cmd).returncode)
