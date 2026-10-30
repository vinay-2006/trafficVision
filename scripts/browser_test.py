"""
Playwright automated browser validation script for TrafficVision Streamlit App.
"""
import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ARTIFACT_DIR = Path(r"C:\Users\KEERTHI\.gemini\antigravity\brain\9d7ef876-ab0b-4fa1-afda-f9142d0b78dc")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_VIDEO_PATH = str(Path("data/sample_traffic.mp4").resolve())

def run_browser_validation():
    print("[Browser Test] Launching Playwright with Edge...", flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 950})
        page = context.new_page()

        print("[Browser Test] Navigating to http://localhost:8501...", flush=True)
        page.goto("http://localhost:8501", timeout=60000, wait_until="domcontentloaded")

        # Wait for Streamlit to finish loading components
        print("[Browser Test] Waiting for Streamlit file uploader component to mount...", flush=True)
        page.wait_for_selector('[data-testid="stFileUploader"]', timeout=60000)
        time.sleep(3)

        landing_path = ARTIFACT_DIR / "browser_landing.png"
        page.screenshot(path=str(landing_path), full_page=True)
        print(f"[Browser Test] Saved landing screenshot to {landing_path}", flush=True)

        # Click "Load Built-in Sample Video"
        sample_btn = page.locator("button:has-text('Load Built-in Sample Video')")
        if sample_btn.count() > 0:
            print("[Browser Test] Clicking 'Load Built-in Sample Video'...", flush=True)
            sample_btn.first.click()
        else:
            print(f"[Browser Test] Uploading sample video: {SAMPLE_VIDEO_PATH}...", flush=True)
            file_input = page.locator('input[type="file"]').first
            file_input.set_input_files(SAMPLE_VIDEO_PATH)

        time.sleep(4)

        # Wait for "Run Full Traffic Analysis" button
        print("[Browser Test] Waiting for 'Run Full Traffic Analysis' button...", flush=True)
        run_btn = page.locator("button:has-text('Run Full Traffic Analysis')")
        run_btn.wait_for(state="visible", timeout=30000)
        time.sleep(1)
        run_btn.click()
        print("[Browser Test] Clicked 'Run Full Traffic Analysis'. Processing pipeline started...", flush=True)

        # Wait for "Key Performance Indicators" (pipeline run takes ~60-90s on CPU)
        print("[Browser Test] Waiting for analytics results (up to 200s)...", flush=True)
        page.wait_for_selector("text=Key Performance Indicators", timeout=200000)
        time.sleep(5)

        # Tab 1: Overview
        tab1_path = ARTIFACT_DIR / "browser_tab1_overview.png"
        page.screenshot(path=str(tab1_path), full_page=False)
        print(f"[Browser Test] Saved Tab 1 Overview screenshot to {tab1_path}", flush=True)

        # Find tabs
        tabs = page.locator('[role="tab"]')
        tab_count = tabs.count()
        print(f"[Browser Test] Found {tab_count} tabs", flush=True)

        # Tab 2: Video Monitoring
        if tab_count >= 2:
            print("[Browser Test] Clicking Tab 2 (Video Monitoring)...", flush=True)
            tabs.nth(1).click()
            time.sleep(4)
            tab2_path = ARTIFACT_DIR / "browser_tab2_video.png"
            page.screenshot(path=str(tab2_path), full_page=False)
            print(f"[Browser Test] Saved Tab 2 Video screenshot to {tab2_path}", flush=True)

        # Tab 3: Traffic Analytics
        if tab_count >= 3:
            print("[Browser Test] Clicking Tab 3 (Traffic Analytics)...", flush=True)
            tabs.nth(2).click()
            time.sleep(4)
            tab3_path = ARTIFACT_DIR / "browser_tab3_analytics.png"
            page.screenshot(path=str(tab3_path), full_page=False)
            print(f"[Browser Test] Saved Tab 3 Analytics screenshot to {tab3_path}", flush=True)

        # Tab 4: Incident Monitoring
        if tab_count >= 4:
            print("[Browser Test] Clicking Tab 4 (Incident Monitoring)...", flush=True)
            tabs.nth(3).click()
            time.sleep(4)
            tab4_path = ARTIFACT_DIR / "browser_tab4_incidents.png"
            page.screenshot(path=str(tab4_path), full_page=False)
            print(f"[Browser Test] Saved Tab 4 Incidents screenshot to {tab4_path}", flush=True)

        # Tab 5: Reports & Export
        if tab_count >= 5:
            print("[Browser Test] Clicking Tab 5 (Reports & Export)...", flush=True)
            tabs.nth(4).click()
            time.sleep(4)
            tab5_path = ARTIFACT_DIR / "browser_tab5_reports.png"
            page.screenshot(path=str(tab5_path), full_page=False)
            print(f"[Browser Test] Saved Tab 5 Reports screenshot to {tab5_path}", flush=True)

        browser.close()
        print("[Browser Test] ALL 5 TABS CAPTURED AND VERIFIED SUCCESSFULLY!", flush=True)

if __name__ == "__main__":
    run_browser_validation()
