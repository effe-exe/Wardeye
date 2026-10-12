# Releasing Wardeye

Two milestones: the repository goes public, and the extension goes to the Chrome Web Store as a free alpha while Riot reviews the application ([D-025](decisions.md#d-025-release-on-the-chrome-web-store-now-and-apply-to-riot-in-parallel)).

## A. Making the repository public

**Done on 29 September 2026:** the repository is public, with its About, topics, social preview and private vulnerability reporting (steps 1–5 below). Still to do: the full review and the trademark search.

Before:
- [x] The checks pass: `npm run check`, `npm run test:e2e` and `cd ml && pytest -q` (the same jobs as CI, green since run 79).
- [ ] The pre-publication review is done and its fixes are in: the extension's security and privacy, the engine, and the licences of everything the builds ship.
- [x] The history is clean (checked on 29 September 2026, all commits): no secrets or keys, no personal email addresses (commits use no-reply addresses), no players' names, no media, weights or archives, no machine paths, no card text or card images.
- [ ] The name: a trademark search for "Wardeye" ([D-020](decisions.md#d-020-the-product-is-called-wardeye)), the Maintainer's call. A web search on 29 September 2026 found no mark or product called Wardeye; the nearest are the US marks WARD and WARDER, WARDIX (binoculars with a camera) and two security extensions, Ward and Warden. A search of the registers (EUIPO or TMview, USPTO, WIPO) in classes 9, 41 and 42 is still to do.

Then, on GitHub:
1. **Settings → General → Danger Zone → Change visibility → Public.**
2. **Actions:** turn the CI workflow back on if it was paused; public repositories run it for free.
3. **About** (the gear next to About on the repository's page): description `Place the ward. See the table.`; topics without Riot's names (for example `computer-vision`, `browser-extension`, `chrome-extension`, `webgpu`, `onnx`, `twitch`); the website once the page is live.
4. **Settings → General → Social preview:** upload `social-preview.png`, rendered by `node scripts/brand-social.mjs OUT` (1280 × 640).
5. **Settings → Code security → Private vulnerability reporting:** on. [SECURITY.md](../SECURITY.md) sends reports there.
6. **The CLA bot** ([.github/workflows/cla.yml](../.github/workflows/cla.yml)) keeps its signatures on the `cla-signatures` branch; it needs Actions to have read and write access (it asks for `contents: write` and `pull-requests: write`).
7. **Branch protection** for `main` (pull requests and passing checks) once other people contribute.
8. Add the repository's link to the web page ([brief](wardeye-page-brief.md), section 8).

## B. The Chrome Web Store release

**Live:** [Wardeye on the Chrome Web Store](https://chromewebstore.google.com/detail/wardeye/hjglackjofehdfecoeehbdmbobafbjhn), item `hjglackjofehdfecoeehbdmbobafbjhn`. Version 0.1.2 passed its first review. The first update is 0.2.1 (below): 0.2.0 was taken out of review before it was published, for the tracker fixes found on the Barcelona final. 0.2.2 was never submitted; 0.2.3 (below) carries its changes and is live since 6 October 2026.

Now, while Riot reviews the application ([D-025](decisions.md#d-025-release-on-the-chrome-web-store-now-and-apply-to-riot-in-parallel)). The store build:
- is standalone only: no companion mode and no access to `127.0.0.1`;
- carries the models ([D-022](decisions.md#d-022-free-for-everyone-closed-weights-a-showcase-for-gradeon)) and the embedding index keyed by printing id, and no card image or card text ([D-015](decisions.md#d-015-no-riot-api-no-riot-assets-distributed)): names, types and images load from Riot's public card gallery as the viewer watches;
- is one zip with the manifest at its root, as the store takes it;
- links this [privacy policy](PRIVACY.md) at a public URL (the file on GitHub once the repository is public).

### The listing

- **Name:** Wardeye
- **Summary** (132 characters at most, the manifest's description): Place the ward. See the table. Wardeye names the cards on the table of Riftbound streams on Twitch, on your own computer.
- **Category:** Lifestyle → Entertainment.
- **Description:**

  > Wardeye is the ward you place on a Riftbound stream. Point at any card on the table to see its name and official image, as the video plays, live or on replay.
  >
  > - Names the cards face up on the table as they are played.
  > - Shows the card when you point at it, and what lies under it, such as gear on a unit.
  > - Works in theatre mode and fullscreen. Turn it off and on with the power button on its badge, its toolbar button, or Alt+R (Option+R on a Mac).
  >
  > Private by design. The recognition runs on your computer, in your browser. No video leaves it, and Wardeye keeps no history. Card names and images load from Riot's public card gallery. It reads only the table camera, never hand cams, face-down cards or the broadcast's hand lists. It is for watching, not playing: no player stats, no win rates.
  >
  > Free for everyone: no ads, no paid features, no account. Open source (AGPL). Alpha: results are published with every model.
  >
  > Wardeye is a community project by Federico Vietti. Wardeye was created under Riot Games' "Legal Jibber Jabber" policy using assets owned by Riot Games. Riot Games does not endorse or sponsor this project.

- **Single purpose:** Identify the cards visible on the table in Riftbound videos on Twitch, and show their names and images on the player.
- **Permission justifications:**
  - `www.twitch.tv`: reads the frames of the video being watched and draws the overlay on the player.
  - `offscreen`: runs the recognition engine in a hidden extension document, so it does not slow the page.
  - `content.publishing.riotgames.com` and `cmsassets.rgpub.io`: load the card names, types and images from Riot's public card gallery; the extension ships none of them.
- **Remote code:** none. Every script and WebAssembly file ships in the package; the models are data inside it.
- **Data use:** does not collect or use user data.
- **Images:** `node scripts/brand-social.mjs OUT` renders the store's own images. They are uploaded, not committed.
  - `store-icon.png`, 128 × 128: the mark 96 px tall with 16 px of transparent padding, as the store asks.
  - `store-promo-small.png`, 440 × 280: the lockup and the one-liner. Required; listings without it are shown after those with one.
  - `store-marquee.png`, 1400 × 560: optional.
  - Screenshots, 1 to 5, at 1280 × 800: the overlay on a table, a hover card open, the badge. Take them from the real extension on a real replay. Crop or blur the broadcast's logos, and credit the broadcast in the description.
- **Links:** the web page, [effe-exe.github.io](https://effe-exe.github.io) ([D-027](decisions.md#d-027-wardeyes-own-page-at-effe-exegithubio)), as the homepage, and the repository's issues page for support. The privacy policy has a field of its own: [PRIVACY.md](PRIVACY.md) at its public URL.

### The 0.2.1 update

The same listing, with these changes:

- **Package:** the 0.2.1 store zip, built and checked as above (the embedder in both precisions). The version must be higher than the one live. 0.2.1 is 0.2.0 with the tracker fixes found on the Barcelona final: a card turned sideways read once as a battlefield is named again as what it is, a battlefield moved or outlined twice is drawn once, and a card held across the table's edge is never read. A card held in a hand over the table is read only once it is put down ([D-029](decisions.md#d-029-a-card-in-a-hand-is-not-on-the-table)). It also takes the table's layout as a guide ([D-028](decisions.md#d-028-the-tables-layout-is-a-guide-not-a-rule)): battlefields are pinned only in the strip along the midline, the Match tab counts each player's runes, stacked ones included, and the exhausted ones, and two pasted lists help each other. The listing, the permissions and their justifications are 0.2.0's.
- **Description**, for the new panel and settings:

  > Wardeye is the ward you place on a Riftbound stream. Point at any card on the table to see its name and official image, as the video plays, live or on replay.
  >
  > - Marks the cards face up on the table and names them as they are played.
  > - Shows the card when you point at it, how sure it is, and what lies under it, such as gear on a unit.
  > - A panel beside the video: each player's legend and the cards on their side, and the plays of the match. On a replay, click a play to jump to it.
  > - Paste a player's published decklist, and Wardeye looks only among those cards.
  > - Choose what stays on the video: outlines and names, outlines only, or a clean view that shows a card only when you point at it. Choose how hard it works, for a computer busy with other apps.
  > - Works in theatre mode and fullscreen. Turn it off and on with the power button on its badge, its toolbar button, or Alt+R (Option+R on a Mac).
  >
  > Private by design. The recognition runs on your computer, in your browser. No video leaves it, and Wardeye keeps no history: it stores only your two settings. Card names and images load from Riot's public card gallery. It reads only the table camera, never hand cams, face-down cards or the broadcast's hand lists. It is for watching, not playing: no player stats, no win rates.
  >
  > Free for everyone: no ads, no paid features, no account. Open source (AGPL). Alpha: results are published with every model.
  >
  > Wardeye is a community project by Federico Vietti. Wardeye was created under Riot Games' "Legal Jibber Jabber" policy using assets owned by Riot Games. Riot Games does not endorse or sponsor this project.

- **Permission justifications** for the two new permissions; neither shows a warning, so an update does not disable the extension for anyone:
  - `sidePanel`: shows Wardeye's panel in the browser's side panel beside the Twitch page: each player's legend and face-up cards, the plays of the match, the decklists the viewer pastes, and the settings.
  - `storage`: remembers the viewer's two settings, what Wardeye shows on the video and how often it reads the video, on their computer (`chrome.storage.local`). Nothing else is stored, and nothing is sent anywhere.
- **Unchanged:** the single purpose, the host permissions, no remote code, no data collected.

### The 0.2.2 update

The same listing as 0.2.1, with a new package:

- **Package:** the 0.2.2 store zip, built and checked as above (the embedder in both precisions). 0.2.2 is 0.2.1 with the fixes found on a co-stream of a Riftbound tournament, recorded live:
  - What a broadcast lays over every shot (a co-streamer's webcam and chat, a scoreboard, a banner) is found from the cuts and never read. A play read on it before it was found is taken off the panel's list ([D-030](decisions.md#d-030-what-stays-put-through-the-cuts-is-the-broadcasts-overlay)).
  - The table is found only in a view of both sides of a table, never in a close-up or a graphic of cards ([D-031](decisions.md#d-031-a-table-camera-shows-both-sides-of-a-table)).
  - A box bigger than a card, or a zone printed on the mat, is not a card, and four cards named at once are not four plays ([D-032](decisions.md#d-032-a-card-is-a-cards-size-and-not-the-mats-print)).
  - On a wooden table the cards beside the wood are read: a hand is what is not the table ([D-033](decisions.md#d-033-a-hand-is-what-is-not-the-table)).
  - The table camera is told by the table window alone, block by block: a close-up of a hand is not taken for it, an arm over the mat no longer takes it away, and a player cam is never learnt as the table ([D-034](decisions.md#d-034-the-table-camera-is-told-by-the-table-window-block-by-block)).
- **Unchanged:** the description, the single purpose, the permissions and their justifications, no remote code, no data collected.
- **On GitHub too:** the same zip goes on the release `v0.2.2` ([D-035](decisions.md#d-035-each-releases-zip-is-on-github-too)), with notes saying what changed and that the models inside are not under the AGPL.

### The 0.2.3 update

The same listing as 0.2.2, with a new package:

- **Package:** the 0.2.3 store zip, built and checked as above (the embedder in both precisions). 0.2.3 is 0.2.2 with a tracker fix and a calmer page:
  - A card a hand has split into two outlines is drawn once and is no longer announced as a play, and is not taken for a card that moved ([D-036](decisions.md#d-036-a-second-outline-of-a-card-is-no-play)).
  - A little less work on the page while Wardeye is off or the video is paused.
- **Unchanged:** the description, the single purpose, the permissions and their justifications, no remote code, no data collected.
- **On GitHub too:** the same zip goes on the release `v0.2.3` ([D-035](decisions.md#d-035-each-releases-zip-is-on-github-too)), with notes saying what changed and that the models inside are not under the AGPL.

### The 0.3.0 update

0.3.0 adds YouTube ([D-037](decisions.md#d-037-youtube-is-asked-for-with-a-click-and-read-only-on-riftbound-videos)). The listing changes in four places.

- **Summary** (the manifest's description, 129 characters): Place the ward. See the table. Wardeye names the cards on the table of Riftbound streams on Twitch and YouTube, on your computer.
- **Description:** the first line becomes "Wardeye is the ward you place on a Riftbound stream, on Twitch or YouTube." Add this bullet after "Works in theatre mode and fullscreen":

  > - On YouTube, click Wardeye's toolbar button once to allow it there. It then starts by itself on videos whose title names Riftbound; turn it on for any other video with the toolbar button or Alt+R.

- **Single purpose:** Identify the cards visible on the table in Riftbound videos on Twitch and YouTube, and show their names and images on the player.
- **Permission justifications** for what is new; none shows a warning, so the update leaves Wardeye on for everyone:
  - `scripting`: runs the overlay on YouTube pages once the user has allowed YouTube, including in a YouTube tab that was already open.
  - `activeTab`: when the user clicks the toolbar button, tells the extension that the tab is a YouTube page, so that it can ask for YouTube.
  - `www.youtube.com` (optional host permission): reads the frames of the YouTube video being watched and draws the overlay on the player. It is asked for only when the user clicks the toolbar button on YouTube, and can be taken back in the browser's settings.
- **Privacy policy:** the new version ([PRIVACY.md](PRIVACY.md), 12 October 2026) says where it runs on YouTube and what it reads there. It is the same URL; nothing to change in the dashboard.
- **Unchanged:** the name, the category, no remote code, no data collected.
- **Riot:** the registration says Twitch only. Before 0.3.0 is published, add to the product's description in the Developer Portal: "From version 0.3, also on YouTube, once the user allows it, and only on Riftbound videos unless the user turns it on" ([riot-application.md](riot-application.md)).

**Before submitting, check it on real YouTube** (about 10 minutes; the browser tests use a stand-in page):

1. Load the 0.3.0 zip unpacked, then open a Riftbound video, e.g. [Riftbound National Chengdu, Day 2 Top Cut](https://www.youtube.com/watch?v=xtVH9iBSni4). Nothing should show yet.
2. Click Wardeye's toolbar button. The browser asks whether Wardeye may read YouTube: allow it. The badge appears on the player, without a reload.
3. Let it find the table. The cards are boxed on the video, not beside or under it, in the normal view, theatre mode and fullscreen.
4. If an ad plays, the overlay goes away during it and comes back after.
5. Open another video from the side list. The plays panel starts over for it.
6. Open a video whose title does not say Riftbound: nothing shows. The toolbar button turns Wardeye on for it, and off again.
7. Go to YouTube's home page and search results. Wardeye shows nothing there, even over the previews that play.
8. Reload a YouTube video: Wardeye comes back by itself. Then take YouTube's access back (`chrome://extensions` → Wardeye → Details → Site access) and reload: Wardeye is gone.
9. Twitch still works as before.

### Downloaded models (0.4.0, not released yet)

From 0.4.0 the store zip holds no model ([D-038](decisions.md#d-038-the-models-are-downloaded-the-first-time-not-shipped-in-the-zip)): 15.6 MB instead of 192 MB. Before the first such release:

1. **The host is a Hugging Face model repository** (D-038), for example `effe-exe/wardeye-models`. Once:
   - Create it, public, at [huggingface.co/new](https://huggingface.co/new).
   - Upload its card: the top of [model-card.md](model-card.md) as `README.md`, the part after the line as `LICENSE.md`.
   - Upload the three model files, exactly as the pack lists them, from `~/rifteye-data/models/onnx/` or from the `models/` folder of the 0.3.0 zip (the same bytes). In the browser: **Files** → **Add file** → **Upload files**. Or from a terminal, after `pip install -U huggingface_hub` and `hf auth login` (a token with write access):

     ```bash
     hf upload OWNER/wardeye-models models/ . --commit-message "detector-v0 and embedder-v1"
     ```

   - Take the commit's full hash from the repository's **History** (or `git ls-remote https://huggingface.co/OWNER/wardeye-models refs/heads/main`). The address names that commit, so a file can never change under it.
2. **Pack with the host's address**, naming a fixed version (a commit), so that a file never changes under its address:

   ```bash
   RIFTEYE_DATA=~/rifteye-data npm run pack:store -w @rifteye/extension -- --out wardeye-0.4.0.zip \
     --precisions detector:fp32,embedder:fp32+fp16 --models-url https://huggingface.co/OWNER/REPO/resolve/COMMIT/
   ```

   The pack prints the files to upload, with their sizes and sha256.
3. **Upload those files, unchanged**, then check that each address serves them: `curl -sI -H 'Origin: chrome-extension://x' URL` shows `200` (or a redirect) and an `access-control-allow-origin` header, and `curl -sL URL | sha256sum` gives the hash the pack printed.
4. **Test on a fresh profile:** load the zip unpacked, open a Riftbound video, and watch the badge say "downloading ..., the first time only" and then read the table. Reload: no download the second time.
5. **Listing:** add the `unlimitedStorage` justification ("keeps the recognition models, downloaded once, without the browser clearing them") and say in the description that the first use downloads about 200 MB once. The privacy policy already says so.

