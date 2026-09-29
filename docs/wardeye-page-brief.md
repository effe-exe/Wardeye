# Brief: the Wardeye page on gradeon.ai

**For:** whoever builds the page in Framer. **URL:** `https://gradeon.ai/wardeye`. **Language:** English.

**What Wardeye is.** A free browser extension for people watching Riftbound on Twitch, live or on replay. While the video plays, it recognises the cards lying on the table and names them: point at a card to see its official image and name. Gradeon makes it. It is not out yet.

**Why the page is needed now.** Riot's developer programme reviews Wardeye before it launches. The application needs "a valid URL for verification purposes", with a demo of the product and its user flow. This page is that URL.
- Riot may ask us to prove we own the site. Check that Framer can take whatever they ask for: a verification file at the site root (e.g. `/riot.txt`), a meta tag, or a DNS record.
- The page must be live before the application is sent.

## Look: Gradeon's

The page uses the same tokens as the Gradeon app (its Tailwind theme), in the same "Nothing OS" style: near-black, precise, a little industrial.

| Token | Value | Use |
|---|---|---|
| Background | `#0A0A0A` | Page |
| Surface | `#111111` | Sections |
| Elevated | `#1A1A1A` | Cards, video frame |
| Card | `#222222` | Inner panels |
| Subtle / border | `#333333` / `#1A1A1A` | Rules, borders |
| Muted text | `#666666` | Secondary text |
| Brand | `#6153CC` | Buttons, links, highlights |
| Brand light | `#7B6FD4` | Hover, the overlay's card outlines in visuals |
| Neon accents, sparingly | `#00FF88` `#CCFF00` `#FF6600` `#FF0044` | Small status marks only |

- **Font:** Space Mono (Google Fonts, SIL Open Font License) for headings, labels and numbers. The system sans-serif for body text.
- **Corners:** sharp, 2 px for buttons and badges, 4 px for cards.
- **Motion:** little, and only to show the product (the demo).

## Page structure and copy

Use the copy as written, or shorten it. Don't add claims beyond it (see the rules below).

**1. Hero**
- Eyebrow: `WARDEYE · BY GRADEON`
- Headline: **See every card on the table.**
- Sub: *A free browser extension for Riftbound viewers on Twitch. Point at any card on the table to see what it is, live or on replay.*
- Buttons:
  - `Watch the demo` scrolls to the video.
  - `Coming soon` is a disabled badge, not a download (see the rules).
  - Optional: `Get notified`, only through Gradeon's existing sign-up form and privacy policy.

**2. Demo video (60 to 90 s)**
- Until the video exists, use a placeholder frame with the text "Demo coming soon".
- Host the file in Framer, or on YouTube or Vimeo as unlisted, not on Twitch.
- Caption: *Wardeye naming the cards on a Riftbound broadcast in Chrome.*

**3. How it works** (three steps)
1. **Add Wardeye** to Chrome or Edge.
2. **Open a Riftbound stream or replay** on Twitch.
3. **Point at a card.** Wardeye boxes the cards on the table and shows each one's official image and name. A card under another is listed with it.

**4. What it does**
- Names the cards face up on the table, as they are played.
- Shows the official card image and name when you point at a card.
- Remembers what lies under a card, such as gear on a unit.
- Works in theatre mode and fullscreen. Press Option+R (Alt+R on Windows) to hide it.

**5. Private by design**
- **Runs on your computer.** Recognition happens in your browser, on your GPU. No video leaves your machine.
- **Public information only.** Wardeye reads only the table camera. It never reads hand cams, face-down cards or the hand lists a broadcast shows.

**6. Made for viewers**
- For watching, not playing: it isn't a play aid. It keeps no player stats and no win rates.
- Free for everyone: no ads, no paid features, no account.

**7. From Gradeon**
- *Wardeye comes from Gradeon, the AI card pre-grading app: the same eye for cards, pointed at the table.*
- Link: `gradeon.ai`.
- Once the repository is public, add: *Open source: github.com/effe-exe/wardeye*. It is still private and still named rifteye, so leave the link out for now.

**8. FAQ**
- **Which browsers?** Chrome and Edge on desktop. A recent graphics chip (WebGPU) makes it fast; without one it is slower.
- **Does it cost anything?** No. It is free for everyone.
- **Does it work on YouTube?** Not yet. Twitch first.
- **Is it made by Riot Games?** No. See the notice below.
- **Does it see my hand?** No, and it doesn't read anyone's. Only the cards on the table, which every viewer can see.

**9. Footer notice**, verbatim, clearly visible and easy to find:

> Wardeye was created under Riot Games' "Legal Jibber Jabber" policy using assets owned by Riot Games. Riot Games does not endorse or sponsor this project.

## Rules for the page

These come from Riot's policies for fan projects and developers. Riot reads the page before approving Wardeye.
- **No Riot logos.** No Riot Games, Riftbound or League of Legends logos, anywhere. That includes the video and screenshots: crop or blur a broadcast's logo.
- **No Riot look.** Nothing designed to look like Riot's or Riftbound's own branding, fonts or card frames. Gradeon's look only.
- **Never say** "official", "partner", "approved by Riot", "endorsed" or "Riot-certified". The notice above is the only Riot statement.
- **Riot's names only in running text.** "Riftbound" may describe what Wardeye is for, in the body text. Keep Riot's names out of the URL, the page title, meta keywords, tags and social handles.
- **No download yet.** The public release waits for Riot's approval, so the button says "Coming soon". Reviewers can be sent a build privately on request.
- **Free means free.** No price, no premium tier, no "Gradeon Pro only" features.
- **Footage:** keep clips short and about the tool, not the match. Credit the broadcast in the caption, e.g. "Footage: a Riftbound Regional Qualifier broadcast".

**SEO and meta:**
- Title: `Wardeye · See every card on the table · Gradeon`.
- Description: `Wardeye is a free browser extension that names the cards on the table as you watch card-game streams on Twitch.`
- OG image: a clean product shot (the overlay on the table, logos cropped), 1200 × 630.

## Demo video: shot list (60 to 90 s)

Record once the extension shows the Wardeye name and Gradeon's look. Until then it still says "RiftEye" on screen.

| Time | Shot | On-screen text |
|---|---|---|
| 0–5 s | Title card, Gradeon look | **Wardeye**: see every card on the table |
| 5–15 s | A Riftbound replay on Twitch in theatre mode. The Wardeye badge reads "finding the table", then boxes appear | *Open any Riftbound stream or replay* |
| 15–45 s | Point at three or four cards: a unit (official image and name), a card with gear under it ("Under it: …"), a legend | *Point at a card to see what it is* |
| 45–60 s | Fullscreen; Option+R hides and shows the overlay | *Works in theatre mode and fullscreen* |
| 60–75 s | Close-up of the badge and a plain table | *Runs on your computer · Reads only the table: never hands or face-down cards* |
| 75–85 s | End card | *Free · by Gradeon · gradeon.ai/wardeye*, then the notice text |

Record at 1080p, with the laptop plugged in and nothing else heavy running. Crop out, or blur, any logo in the broadcast's graphics.

## Checklist before the page goes live

- [ ] URL `gradeon.ai/wardeye`; title and meta as above
- [ ] Gradeon tokens and Space Mono; dark page; sharp corners
- [ ] All nine sections; "Coming soon" instead of a download
- [ ] The notice, verbatim, easy to find
- [ ] No Riot logos or Riot-like design; none of the forbidden words; no Riot names in URL, title, keywords or tags
- [ ] Demo video in place, or the placeholder until it is recorded
- [ ] A way to add Riot's ownership check (a file at the root, a meta tag, or DNS)
