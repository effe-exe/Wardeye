# Wardeye privacy policy

**Last updated: 29 September 2026.** Wardeye is a free, open-source browser extension, a community project by Federico Vietti. This policy covers the extension as published on the Chrome Web Store. Its code is public, so everything below can be checked in [apps/extension](../apps/extension/).

## In short

Wardeye collects nothing. It has no account, no server, no analytics, no ads and no tracking. It reads the Twitch video you are watching on your own computer and draws its results on the player. The only thing it loads from the internet is card data from Riot Games' public card gallery.

## What it reads

- **Only on twitch.tv.** The extension runs on `https://www.twitch.tv/*` pages and nowhere else.
- **The frames of the video you are watching**, while it plays, to find and name the cards on the table. Only the table camera's area is analysed: hand cams, face-down cards and the broadcast's side graphics are never processed. When the video is paused, nothing is read.
- **The video's address on twitch.tv** (for example `/videos/12345`), so that each video gets its own board. It stays in the extension.

## Where it is processed

In the extension itself, in your browser, on your graphics chip or processor. The recognition models ship inside the extension. No frame, and nothing learnt from one, ever leaves your computer.

## What it loads from the internet

Card names, types and images, from Riot Games' public card gallery: the gallery's card list from `content.publishing.riotgames.com`, and each card's image from `cmsassets.rgpub.io` when it is shown. These are ordinary web requests, like visiting the gallery: Riot's servers see your IP address and your browser, and a request for a card's image tells them which card is being shown. Riot's own privacy policy applies to them. Wardeye sends nothing else, to Riot or to anyone.

## What it stores

Nothing of its own: no cookies, no history of what you watched, no settings about you. When you close the tab, the board is gone. Your browser may keep the card list and the card images in its ordinary cache, as it does for any web page.

## The permissions it asks for

| Permission | Why |
|---|---|
| Read and change data on `www.twitch.tv` | To read the video frames and draw the overlay on the player |
| `offscreen` | To run the recognition engine in a hidden document of the extension, so it does not slow the page |
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
