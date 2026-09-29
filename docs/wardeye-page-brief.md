# Brief: the Wardeye page

**For:** whoever builds the page in Framer, and their agent. **URL:** `https://gradeon.ai/wardeye`. **Language:** English.

**Read first:** the brand book, [assets/brand/README.md](../assets/brand/README.md). It holds the logo files, colour tokens, fonts, voice and visual language this brief uses. Everything here follows it.

**What Wardeye is.** A free browser extension for people watching Riftbound on Twitch, live or on replay. While the video plays, it recognises the cards lying on the table and names them: point at a card to see its official image and name. It is a community project by Federico Vietti, who also makes Gradeon. It is not a Gradeon product, and it is not out yet.

**Why the page is needed now.** Riot's developer programme reviews Wardeye before it launches. The application needs "a valid URL for verification purposes", with a demo of the product and its user flow. This page is that URL.
- Riot may ask us to prove we own the site. Check that Framer can take whatever they ask for: a verification file at the site root (e.g. `/riot.txt`), a meta tag, or a DNS record.
- The page must be live before the application is sent.

## Look: the brand book's

Wardeye inherits Gradeon's dark-first system (the same dark base, the same violet family), with its own tokens, logo and type. Match Gradeon's dark layout: a near-black background, violet calls to action, generous space, short sentences.

| Token | Value | Use |
|---|---|---|
| Background | `#0A0A0B` | The page |
| Surface | `#121214` | Sections, cards |
| Surface 2 | `#1A1A1E` | Raised panels, the video frame |
| Border | `#2A2A30` | Hairline borders and rules |
| Text | `#F4F4F5` | Text |
| Muted | `#A1A1AA` | Secondary text |
| Dim | `#71717A` | Captions, the footer |
| Primary | `#8B7CF6` | Buttons, links, focus. Never behind body text |
| Primary light / dark | `#A99BFF` / `#6D5BD0` | Hover / pressed |
| Accent | `#C4B5FD` | Soft emphasis |
| Success, warning, error | `#34D399`, `#FBBF24`, `#F87171` | Small status marks only; warning is the alpha badge |

- **Type** (all on Google Fonts, SIL Open Font License): **Space Grotesk** for the hero and section titles (Bold 28–36 for the hero, SemiBold 22 for sections), **Inter** for everything else (body 14–16, captions 11–13), **JetBrains Mono** for numbers (results, frame rates). Section labels: Inter SemiBold 11 px, in capitals, in primary.
- **Logo:** the lockup, `assets/brand/logo/lockup.svg`, in the hero; the mark alone (`mark.svg`) as the favicon and in the footer. Clear space around the lockup equals the mark's height. Never stretch, recolour or add effects to it.
- **Shape and motion:** soft corners (6 px for buttons and badges, 8 px for cards, 12 px for large panels), hairline borders, little motion (150–250 ms ease-out on hover), none of it continuous.

## Page structure and copy

Use the copy as written, or shorten it. Don't add claims beyond it (see the rules below). The voice is precise and calm: numbers over adjectives, no hype.

**1. Hero**
- The lockup, then the eyebrow `ALPHA · A COMMUNITY PROJECT`
- Headline: **Place the ward. See the table.**
- Sub: *Wardeye is the ward you place on a Riftbound stream. Hover any card to inspect it, follow the match as a timeline, and see the whole board at a glance, from the video alone.*
- Buttons:
  - `Watch the demo` (primary) scrolls to the video.
  - `Coming soon` is a disabled badge, not a download (see the rules).
  - Optional: `Get notified`, only through an existing sign-up form with its privacy policy.

**2. Demo video (60 to 90 s)**
- Until the video exists, a placeholder frame (Surface 2, the mark centred) with the text "Demo coming soon".
- Host the file in Framer, or on YouTube or Vimeo as unlisted, not on Twitch.
- Caption: *Wardeye naming the cards on a Riftbound broadcast in Chrome. Footage: a Riftbound Regional Qualifier broadcast.*

**3. How it works** (three steps)
1. **Place the ward.** Add Wardeye to Chrome or Edge.
2. **Open a Riftbound stream or replay** on Twitch.
3. **Point at a card.** Wardeye outlines the cards on the table and shows each one's official image and name. A card under another is listed with it.

**4. What it does**
- Names the cards face up on the table as they are played.
- Shows the official card image and name when you point at a card.
- Remembers what lies under a card, such as gear on a unit.
- Works in theatre mode and fullscreen. Option+R (Alt+R on Windows) hides it.

**5. Measured, not claimed** (numbers in JetBrains Mono)
- `97.6%` of the cards found, on 5,465 reviewed cards from three broadcasts.
- `96.0%` and `99.4%` of the cards named, on two broadcasts held out of training.
- Label under the numbers: *Alpha results, measured on real broadcasts. The reports are public on GitHub* (link once the repository is public).

**6. Private by design**
- **Runs on your computer.** Recognition happens in your browser, on your GPU. No video leaves your machine.
- **Public information only.** Wardeye reads only the table camera. It never reads hand cams, face-down cards or the hand lists a broadcast shows.

**7. Made for viewers**
- For watching, not playing: it isn't a play aid. It keeps no player stats and no win rates.
- Free for everyone: no ads, no paid features, no account.

**8. Who makes it**
- *Wardeye is a community project by Federico Vietti, who also makes [Gradeon](https://gradeon.ai), the AI card pre-grading app. It shares Gradeon's dark look; it is not a Gradeon product.*
- Once the repository is public, add: *Open source (AGPL): github.com/effe-exe/wardeye*. It is still private, so leave the link out for now.

**9. FAQ**
- **Which browsers?** Chrome and Edge on desktop. A recent graphics chip (WebGPU) makes it fast; without one it is slower.
- **Does it cost anything?** No. It is free for everyone.
- **Does it work on YouTube?** Not yet. Twitch first.
- **Is it made by Riot Games?** No. See the notice below.
- **Is it a Gradeon product?** No. It is a community project by the person who makes Gradeon.
- **Does it see my hand?** No, and it doesn't read anyone's. Only the cards on the table, which every viewer can see.

**10. Footer notice**, verbatim, clearly visible and easy to find:

> Wardeye was created under Riot Games' "Legal Jibber Jabber" policy using assets owned by Riot Games. Riot Games does not endorse or sponsor this project.

## Rules for the page

These come from Riot's policies for fan projects and developers, and from the brand book. Riot reads the page before approving Wardeye.
- **No Riot logos.** No Riot Games, Riftbound or League of Legends logos, anywhere. That includes the video and screenshots: crop or blur a broadcast's logo.
- **No Riot look.** Nothing designed to look like Riot's or Riftbound's own branding, fonts or card frames. The brand book's look only.
- **Never say** "official", "partner", "approved by Riot", "endorsed" or "Riot-certified". The notice above is the only Riot statement.
- **Riot's names only in running text.** "Riftbound" may describe what Wardeye is for, in the body text. Keep Riot's names (and champion names) out of the URL, the page title, headings, meta keywords, tags and social handles.
- **Not a Gradeon product.** The page lives on gradeon.ai, but Wardeye is the maker's community project: no "by Gradeon", no Gradeon logo next to the Wardeye lockup, no "Gradeon" in the page title. The credit in section 8 is the only mention.
- **No download yet.** The public release waits for Riot's approval, so the button says "Coming soon". Reviewers can be sent a build privately on request.
- **Free means free.** No price, no premium tier, no paid features.
- **Measured means published.** Only the numbers in section 5, which come from the repository's reports.
- **Footage:** keep clips short and about the tool, not the match. Credit the broadcast in the caption.

**SEO and meta:**
- Title: `Wardeye · Place the ward. See the table.`
- Description: `Wardeye is a free browser extension that names the cards on the table as you watch card-game streams on Twitch. It runs on your computer; no video leaves it.`
- OG image: 1200 × 630, the background colour, the lockup and the one-liner (the brand book's social preview). A clean product shot, logos cropped, also works.
- Favicon: `assets/brand/logo/mark.svg`.

## Demo video: shot list (60 to 90 s)

Record once the extension shows the Wardeye name and the brand book's look.

| Time | Shot | On-screen text (Space Grotesk titles, Inter captions) |
|---|---|---|
| 0–5 s | Title card: the background colour, the lockup | **Place the ward. See the table.** |
| 5–15 s | A Riftbound replay on Twitch in theatre mode. The Wardeye badge reads "finding the table", then outlines appear | *Open any Riftbound stream or replay* |
| 15–45 s | Point at three or four cards: a unit (official image and name), a card with gear under it ("Under it: …"), a legend | *Point at a card to see what it is* |
| 45–60 s | Fullscreen; Option+R hides and shows the overlay | *Works in theatre mode and fullscreen* |
| 60–75 s | Close-up of the badge and a plain table | *Runs on your computer · Reads only the table: never hands or face-down cards* |
| 75–85 s | End card: the lockup | *Free · Alpha · a community project by Federico Vietti*, then the notice text |

Record at 1080p, with the laptop plugged in and nothing else heavy running. Crop out, or blur, any logo in the broadcast's graphics.

## Checklist before the page goes live

- [ ] URL `gradeon.ai/wardeye`; title and meta as above; the mark as favicon
- [ ] The brand book's tokens, fonts and logo files; dark page; soft corners; hairline borders
- [ ] All ten sections; "Coming soon" instead of a download
- [ ] The notice, verbatim, easy to find
- [ ] No Riot logos or Riot-like design; none of the forbidden words; no Riot names in the URL, title, headings, keywords or tags
- [ ] Not presented as a Gradeon product: the credit in section 8 only
- [ ] Demo video in place, or the placeholder until it is recorded
- [ ] A way to add Riot's ownership check (a file at the root, a meta tag, or DNS)
