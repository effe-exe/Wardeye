# Releasing Wardeye

Two milestones: the repository goes public, and the extension goes to the Chrome Web Store as a free alpha while Riot reviews the application ([D-025](decisions.md#d-025-release-on-the-chrome-web-store-now-and-apply-to-riot-in-parallel)).

## A. Making the repository public

Before:
- [ ] The checks pass: `npm run check`, `npm run test:e2e` and `cd ml && pytest -q` (the same jobs as CI).
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
  > - Works in theatre mode and fullscreen. Alt+R (Option+R on a Mac) hides it.
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
  - Screenshots, 1 to 5, at 1280 × 800: the overlay on a table, a hover card open, the badge. Take them from the real extension on a real replay (see below). Crop or blur the broadcast's logos, and credit the broadcast in the description.
- **Links:** the web page once it is live; until then, the public repository as the homepage and its issues page for support. The privacy policy is its own field (below).

### Uploading, step by step

**First, try the zip on your own computer.** The store installs exactly what is in it.
1. Unzip `wardeye-VERSION.zip` into a folder.
2. In Chrome, open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and choose that folder.
3. Open a Riftbound replay on Twitch in theatre mode, for example the Los Angeles grand final at `https://www.twitch.tv/videos/2885620401?t=14h54m00s`. Twitch deletes old broadcasts, so check that it still plays.
   - The badge on the player says it is finding the table, then how many cards it has named.
   - Point at a card: the hover card shows its name and image.
   - Alt+R (Option+R on a Mac) hides and shows the overlay.
4. Take the screenshots now, at 1280 × 800.
5. Remove the unpacked copy on `chrome://extensions` before you install the store version.

**Then, in the [Developer Dashboard](https://chrome.google.com/webstore/devconsole):**
1. **Add new item** → **Choose file** → the zip → **Upload**. The name, summary, version and icons come from the manifest.
2. **Store listing:**
   - Description: the one above.
   - Category: Lifestyle → Entertainment. Language: English.
   - Graphic assets: the store icon, the screenshots, the small promo tile, and the marquee if you like. No video until the demo exists.
   - Homepage URL: the repository, later the web page. Support URL: the repository's issues page.
   - Leave the official URL empty: it needs a domain verified in Google Search Console.
   - Mature content: no.
3. **Privacy:**
   - Single purpose: the one above.
   - A justification for each permission: `offscreen` and the host permissions (`www.twitch.tv`, `content.publishing.riotgames.com`, `cmsassets.rgpub.io`), as above.
   - Remote code: **No**. Everything that runs ships in the zip.
   - Data usage: tick no data type, and tick the certifications (no selling or transfer of user data, no use unrelated to the single purpose, no use for creditworthiness or lending).
   - Privacy policy: `https://github.com/effe-exe/Wardeye/blob/main/docs/PRIVACY.md`.
4. **Distribution:** free; all regions.
   - Visibility **Public** lists it in the store.
   - **Unlisted** is the quieter start: anyone with the link can install it, but it is not in search.
   - **Private** is for named testers only.
   - Every visibility goes through the same review.
5. **Test instructions**, for the reviewer: *No account needed. Open a Riftbound replay on Twitch (for example the link above) and let it play for 20 seconds in theatre mode. The Wardeye badge on the player says it is finding the table, then outlines appear on the cards. Point at a card to see its name and image. Alt+R hides the overlay. It runs on WebGPU where the computer has it, otherwise on the processor, more slowly.*
6. **Submit for review** and confirm.
   - To choose the moment it goes live yourself, untick the option to publish automatically after the review. An approved item then waits up to 30 days for you to publish it.
7. **The review** usually takes a few days and can take a few weeks. New developers get a closer look, and so do extensions with a lot of code. A rejection comes by email with the policy it cites, and a fixed version can be resubmitted.

**After it is live:** put the listing's link in the README's Install section, and in the web page's `Add to Chrome` button ([brief](wardeye-page-brief.md)). Add it to the Riot application too.
