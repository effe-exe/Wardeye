# Wardeye privacy policy

**Last updated: 29 September 2026.** Wardeye is a free, open-source browser extension, a community project by Federico Vietti. This policy covers the extension as published on the Chrome Web Store. Its code is public, so everything below can be checked in [apps/extension](../apps/extension/).

## In short

Wardeye collects nothing. It has no account, no server, no analytics, no ads and no tracking. It reads the Twitch video you are watching on your own computer and draws its results on the player. The only thing it loads from the internet is card data from Riot Games' public card gallery.

## What it reads

- **Only on twitch.tv.** The extension runs on `https://www.twitch.tv/*` pages and nowhere else.
- **The frames of the video you are watching**, while it plays, to find and name the cards on the table. Only the table camera's area is analysed: hand cams, face-down cards and the broadcast's side graphics are never processed. When the video is paused, or you turn Wardeye off (its power button, its toolbar button, or Alt+R, Option+R on a Mac), nothing is read.
- **The video's address on twitch.tv** (for example `/videos/12345`), so that each video gets its own board. It stays in the extension.

## Where it is processed

In the extension itself, in your browser, on your graphics chip or processor. The recognition models ship inside the extension. No frame ever leaves your computer, and neither does what Wardeye learns from one, with one exception: to show a card's picture, it asks Riot's image server for that card's image, which tells Riot which card is being shown (below).

## What it loads from the internet

Card names, types and images, from Riot Games' public card gallery:

- **The card list**, from `content.publishing.riotgames.com`, when you open a page on twitch.tv, whether or not a video plays. It is loaded again when the extension restarts its engine, for example after five minutes without a video.
- **A card's image**, from `cmsassets.rgpub.io`, when the overlay shows that card.

These are ordinary web requests, like visiting the gallery. Riot's servers see your IP address, your browser, and that the request comes from the Wardeye extension (the browser names the extension as the request's origin). A request for a card's image also tells them which card is being shown. The requests carry no cookies and no referrer. Riot's own privacy policy applies to them. Wardeye sends nothing else, to Riot or to anyone.

## What it stores

Nothing of its own: no cookies, no history of what you watched, no settings about you. When you close the tab, the board is gone. From version 0.2, the plays panel lists the plays of the video in that tab, and holds any decklist you paste into it; both are kept only in the tab's memory and go when the tab closes or another video starts. A pasted decklist is read on your computer and sent nowhere. Your browser may keep the card list and the card images in its ordinary cache, as it does for any web page.

## The permissions it asks for

| Permission | Why |
|---|---|
| Read and change data on `www.twitch.tv` | To read the video frames and draw the overlay on the player |
| `offscreen` | To run the recognition engine in a hidden document of the extension, so it does not slow the page |
| `sidePanel` (from version 0.2) | To show the plays panel, the plays and each player's side of the table, in the browser's side panel beside the page |
| Access to `content.publishing.riotgames.com` and `cmsassets.rgpub.io` | To load the card names, types and images from Riot's public card gallery |

The extension also makes its three fonts available to twitch.tv pages, for the overlay. A page could use them to tell that Wardeye is installed; they carry no information about you.

## Builds from source

A developer building Wardeye from the repository can also run it in companion mode, where frames go to a program on the same computer (`http://127.0.0.1`). The Chrome Web Store version has no companion mode and no access to `127.0.0.1`.

## Children

Wardeye is not directed at children, and it collects no data from anyone.

## Changes

Changes to this policy are made in the public repository, where every version stays visible in its history, and are noted in the release notes.

## Contact

Open an issue on the repository, or reach the Maintainer through the GitHub profile [@effe-exe](https://github.com/effe-exe). Security problems: see [SECURITY.md](../SECURITY.md).
