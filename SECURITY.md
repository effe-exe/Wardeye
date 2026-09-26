# Security policy

RiftEye has no released versions yet. Once the browser extension ships, the latest store release is the supported version.

## Reporting a vulnerability

Please report vulnerabilities **privately**, using GitHub's private vulnerability reporting: **Security** tab → **Report a vulnerability**. Please do not open public issues for security problems.

Include what you found, how to reproduce it, and what an attacker could do with it. You will get an acknowledgement within a few days and a fix plan once the issue is confirmed.

## Areas that matter most

- The browser extension's permissions, message passing between content script, worker and side panel, and anything that renders catalogue data into the page.
- Integrity of downloaded catalogue and model files: version and hash checks, and the encoder/index guard.
- The opt-in correction upload: what is sent, and whether anything beyond the documented payload could leak.
- Poisoning of the active-learning queue through crafted corrections.
