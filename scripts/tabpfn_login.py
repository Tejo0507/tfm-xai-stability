"""One-time TabPFN-2.5 license login on Windows.

tabpfn's own login polls stdin with select(), which fails on Windows before the browser
callback can arrive. This runs the same callback flow without the stdin prompt, then
saves the token where tabpfn looks for it (~/.cache/tabpfn/auth_token).

    .venv/Scripts/python.exe scripts/tabpfn_login.py
"""

import os
import threading
import urllib.parse
import webbrowser

from tabpfn import browser_auth as b
from tabpfn.settings import settings

HF_REPO = "tabpfn_2_5"


def main() -> None:
    gui, api = settings.tabpfn.auth_gui_url, settings.tabpfn.auth_api_url
    done, received = threading.Event(), [None]
    httpd, port = b._create_callback_server(gui, done, received)
    threading.Thread(target=b._serve_until_event, args=(httpd, done), daemon=True).start()
    url = f"{gui}/login?callback=http://localhost:{port}&hf_repo_id={urllib.parse.quote(HF_REPO)}"
    webbrowser.open(url)
    print(f"Log in and accept the license in the browser window that opened:\n  {url}\n"
          "Waiting for the browser to hand back the token (Ctrl+C to cancel)...")
    while not done.wait(0.5):
        pass
    httpd.server_close()

    token = received[0]
    if not token or b.verify_token(token, api) is not True:
        raise SystemExit("Login did not return a valid token; try again.")
    licensed = b.check_license_accepted(token, api, b._get_license_name(HF_REPO))
    b.save_token(token)
    print(f"Token saved to {b._TOKEN_FILE}. License accepted: {licensed}")
    if os.environ.get("TABPFN_TOKEN"):
        print("Note: TABPFN_TOKEN is set in the environment and takes priority over the saved token.")


if __name__ == "__main__":
    main()
