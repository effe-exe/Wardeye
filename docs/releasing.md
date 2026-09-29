# Releasing Wardeye

Two milestones. The repository goes public first; the store release waits for Riot's answer ([D-021](decisions.md#d-021-apply-to-riot-for-a-riftbound-app-key)).

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

Waits for Riot's approval ([riot-application.md](riot-application.md)). It needs, in the product:
- card data from Riot at display time, so no card image or card text ships in the package ([D-015](decisions.md#d-015-no-riot-api-no-riot-assets-distributed));
- the public in-browser build, with the models inside the package ([D-022](decisions.md#d-022-free-for-everyone-closed-weights-a-showcase-for-gradeon)), lighter where the measurements allow;
- this [privacy policy](PRIVACY.md) at a public URL, updated for the card images loaded from Riot.

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
  > Private by design. The recognition runs on your computer, in your browser. No video leaves it, and Wardeye keeps no history. It reads only the table camera, never hand cams, face-down cards or the broadcast's hand lists. It is for watching, not playing: no player stats, no win rates.
  >
  > Free for everyone: no ads, no paid features, no account. Open source (AGPL). Alpha: results are published with every model.
  >
  > Wardeye is a community project by Federico Vietti. Wardeye was created under Riot Games' "Legal Jibber Jabber" policy using assets owned by Riot Games. Riot Games does not endorse or sponsor this project.

- **Single purpose:** Identify the cards visible on the table in Riftbound videos on Twitch, and show their names and images on the player.
- **Permission justifications:**
  - `www.twitch.tv`: reads the frames of the video being watched and draws the overlay on the player.
  - `offscreen`: runs the recognition engine in a hidden extension document, so it does not slow the page.
  - `http://127.0.0.1`: talks to the optional live runner on the user's own computer (companion mode).
- **Remote code:** none. Every script and WebAssembly file ships in the package; the models are data inside it.
- **Data use:** does not collect or use user data.
- **Images:**
  - the icon (`apps/extension/icons/icon-128.png`);
  - at least one screenshot, 1280 × 800: the overlay on a table, a hover card open, the badge. Logos in the broadcast are cropped or blurred, and the broadcast is credited;
  - a small promo tile, 440 × 280: the lockup on the background colour.
- **Links:** the web page, the public repository, and the privacy policy.
