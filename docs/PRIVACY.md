# Wardeye privacy policy

**Last updated: 12 October 2026.** Wardeye is a free, open-source browser extension, a community project by Federico Vietti. This policy covers the extension as published on the Chrome Web Store and, once it is there, on Firefox Add-ons. Its code is public, so everything below can be checked in [apps/extension](../apps/extension/).

## In short

Wardeye collects nothing. It has no account, no server, no analytics, no ads and no tracking. It reads the Twitch or YouTube video you are watching on your own computer and draws its results on the player. The only things it loads from the internet are card data from Riot Games' public card gallery and, from version 0.4, its own recognition models, once.

## What it reads

- **On twitch.tv, and on youtube.com only if you allow it.** The extension runs on `https://www.twitch.tv/*` pages. From version 0.3 it can also run on `https://www.youtube.com/*` pages, but only after you allow it: the first time you click Wardeye's toolbar button on a YouTube page, your browser asks you. You can take that back at any time in the browser's extension settings. It runs nowhere else.
- **On YouTube, only Riftbound videos, unless you turn it on.** Wardeye reads a YouTube video by itself only when the video's title names Riftbound. It reads any other video only after you turn it on for that video, with its toolbar button or Alt+R. It never reads YouTube's home page, search results, Shorts or the previews of other videos.
- **The frames of the video you are watching**, while it plays, to find and name the cards on the table. Only the table camera's area is analysed: hand cams, face-down cards and the broadcast's side graphics are never processed. When the video is paused, or you turn Wardeye off (its power button, its toolbar button, or Alt+R, Option+R on a Mac), nothing is read.
- **The video's address** (for example `/videos/12345` on Twitch, or the video's id on YouTube) and **its title**, so that each video gets its own board, and so that Wardeye knows a Riftbound video on YouTube. They stay in the extension.

## Where it is processed

In the extension itself, in your browser, on your graphics chip or processor. Up to version 0.3 the recognition models ship inside the extension; from version 0.4 they are downloaded once, the first time Wardeye reads a video, and kept in your browser (below). No frame ever leaves your computer, and neither does what Wardeye learns from one, with one exception: to show a card's picture, it asks Riot's image server for that card's image, which tells Riot which card is being shown (below).

## What it loads from the internet

Card names, types and images, from Riot Games' public card gallery:

- **The card list**, from `content.publishing.riotgames.com`, when you open a page on twitch.tv, whether or not a video plays, and on YouTube when Wardeye starts reading a video. It is loaded again when the extension restarts its engine, for example after five minutes without a video.
- **A card's image**, from `cmsassets.rgpub.io`, when the overlay shows that card.

**Wardeye's models** (from version 0.4), once: the first time it reads a video, it downloads the two files it needs, about 190 to 230 MB, and keeps them in the browser's storage for the extension. It checks that each is exactly the file the extension was made with, and from then on reads them from your computer. A new version of Wardeye may download new models once, and deletes the old ones. The request sends no cookie and no referrer; the host it comes from sees your IP address and that Wardeye asked, as any website does.

The requests to Riot are ordinary web requests, like visiting the gallery. Riot's servers see your IP address, your browser, and that the request comes from the Wardeye extension (the browser names the extension as the request's origin). A request for a card's image also tells them which card is being shown. The requests carry no cookies and no referrer. Riot's own privacy policy applies to them. Wardeye sends nothing else, to Riot or to anyone.

## What it stores

No cookies, no history of what you watched, nothing about you. From version 0.2 it keeps two settings you choose in the plays panel, what it shows on the video (outlines and names, outlines only, or clean) and the performance level (Full, Balanced or Light), in the extension's own storage in your browser; they are never sent anywhere, and go when you remove the extension. When you close the tab, the board is gone. From version 0.2, the plays panel lists the plays of the video in that tab, and holds any decklist you paste into it; both are kept only in the tab's memory and go when the tab closes or another video starts. A pasted decklist is read on your computer and sent nowhere. Your browser may keep the card list and the card images in its ordinary cache, as it does for any web page. From version 0.4 it also keeps Wardeye's downloaded models, which are files of the extension, not data about you; they go when you remove the extension.

## The permissions it asks for

| Permission | Why |
|---|---|
| Read and change data on `www.twitch.tv` | To read the video frames and draw the overlay on the player |
| Read and change data on `www.youtube.com` (optional, from version 0.3; asked only when you click the toolbar button on YouTube) | The same, on YouTube |
| `scripting` (from version 0.3) | To run the overlay on YouTube pages once you allow it, including a YouTube tab that was already open |
| `activeTab` (from version 0.3) | To know, when you click the toolbar button, that the tab is a YouTube page, so that it can ask you for YouTube |
| `unlimitedStorage` (from version 0.4) | To keep the downloaded models (about 190 to 230 MB) without the browser clearing them to make room |
| `offscreen` | To run the recognition engine in a hidden document of the extension, so it does not slow the page |
| `sidePanel` (from version 0.2) | To show the plays panel, the plays and each player's side of the table, in the browser's side panel beside the page |
| `storage` (from version 0.2) | To remember the two settings chosen in the panel (what it shows on the video, and the performance level), on your computer |
| Access to `content.publishing.riotgames.com` and `cmsassets.rgpub.io` | To load the card names, types and images from Riot's public card gallery |

The extension also makes its three fonts available to twitch.tv and youtube.com pages, for the overlay. A page could use them to tell that Wardeye is installed; they carry no information about you.

## Builds from source

A developer building Wardeye from the repository can also run it in companion mode, where frames go to a program on the same computer (`http://127.0.0.1`). The Chrome Web Store version has no companion mode and no access to `127.0.0.1`.

## Children

Wardeye is not directed at children, and it collects no data from anyone.

## Changes

Changes to this policy are made in the public repository, where every version stays visible in its history, and are noted in the release notes.

## Contact

Open an issue on the repository, or reach the Maintainer through the GitHub profile [@effe-exe](https://github.com/effe-exe). Security problems: see [SECURITY.md](../SECURITY.md).
