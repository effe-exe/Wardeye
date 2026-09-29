# Wardeye privacy policy

**Last updated: 29 September 2026.** Wardeye is a free, open-source browser extension, a community project by Federico Vietti. This policy covers the extension and Wardeye's own pages. Its code is public, so everything below can be checked in [apps/extension](../apps/extension/).

## In short

Wardeye collects nothing. It has no account, no server, no analytics, no ads and no tracking. It reads the Twitch video you are watching on your own computer, draws its results on the player, and sends nothing to the internet.

## What it reads

- **Only on twitch.tv.** The extension runs on `https://www.twitch.tv/*` pages and nowhere else.
- **The frames of the video you are watching**, while it plays, to find and name the cards on the table. Only the table camera's area is analysed: hand cams, face-down cards and the broadcast's side graphics are never processed. When the video is paused, nothing is read.
- **The video's address on twitch.tv** (for example `/videos/12345`), so that each video gets its own board.

## Where it is processed

- **In the extension itself** (the standalone build): the recognition runs in your browser, on your graphics chip or processor. The models and the card data ship inside the extension; nothing is downloaded while you watch.
- **Or by a program you run yourself** (companion mode): the extension sends each frame (a JPEG) and the video's address to the Wardeye live runner on the same computer, at `http://127.0.0.1`. It never sends them anywhere else.

## What it stores

Nothing. No cookies, no local storage, no history of what you watched. When you close the tab, the board is gone.

## What it sends over the network

Nothing, beyond your own computer: in companion mode it talks to `127.0.0.1`, which is your machine. The extension makes no other request.

A future build will show the official card images from Riot Games' public card gallery. Your browser will then load those images from Riot's servers, as it would on the gallery itself, and Riot's own privacy policy applies to those requests. This policy will say so before that build ships.

## The permissions it asks for

| Permission | Why |
|---|---|
| Read and change data on `www.twitch.tv` | To read the video frames and draw the overlay on the player |
| `offscreen` | To run the recognition engine in a hidden document of the extension, so it does not slow the page |
| Access to `http://127.0.0.1` | To talk to the live runner on your own computer, in companion mode |

The extension also makes its three fonts available to twitch.tv pages, for the overlay. A page could use them to tell that Wardeye is installed; they carry no information about you.

## Children

Wardeye is not directed at children, and it collects no data from anyone.

## Changes

Changes to this policy are made in the public repository, where every version stays visible in its history, and are noted in the release notes.

## Contact

Open an issue on the repository, or reach the Maintainer through the GitHub profile [@effe-exe](https://github.com/effe-exe). Security problems: see [SECURITY.md](../SECURITY.md).
