"""Stable entry point for the LegalLens research desk."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name('research_app.py')), run_name='__main__')
