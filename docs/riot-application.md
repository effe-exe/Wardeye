# Applying to Riot for a Riftbound app key

**Status: sent on 30 September 2026, in Riot's review.** Wardeye is registered in the Developer Portal as a production product for Riftbound. Its product URL, [effe-exe.github.io](https://effe-exe.github.io), is verified ([D-027](decisions.md#d-027-wardeyes-own-page-at-effe-exegithubio)). The Chrome Web Store alpha went through the store's review at the same time and is [live](https://chromewebstore.google.com/detail/wardeye/hjglackjofehdfecoeehbdmbobafbjhn) ([D-025](decisions.md#d-025-release-on-the-chrome-web-store-now-and-apply-to-riot-in-parallel)), as the application said it would be.

Why apply: Riot's Riftbound policy asks every product that serves players to register, commercial or not, and an approved key is the only way to take card data from the Riot API. Wardeye applies as what it is: a free, non-commercial community project by Federico Vietti ([D-024](decisions.md#d-024-wardeye-has-its-own-brand-book)). It is not a Gradeon product, and it keeps a business out of it, because Riot's fan-content policy (the Legal Jibber Jabber, LJJ) counts "any Project that involves a business or legal entity" as commercial, even a free one.

Until a key is approved, Wardeye stays as it is ([D-015](decisions.md#d-015-no-riot-api-no-riot-assets-distributed), [08 §8.2](research/08-legal-and-policy.md#82-riot-games-policies)). This is a risk analysis, not legal advice. The live pages: [the Riftbound page](https://developer.riotgames.com/docs/riftbound), [the Riftbound policy](https://developer.riotgames.com/policies/riftbound), [product registration](https://developer.riotgames.com/docs/portal), [site verification](https://developer.riotgames.com/how-to-verify-site.html) and [the LJJ](https://www.riotgames.com/en/legal).

The Riot API has one Riftbound content API, `riftbound-content-v1`. Its `GET /riftbound/content/v1/contents` returns the sets and their cards: each card's name, type, text, rarity, faction, stats, keywords and art URLs.

## How it was sent

1. Sign in at [developer.riotgames.com](https://developer.riotgames.com) with the Maintainer's own Riot account.
2. Click **Register Product** and choose the larger-scale (production) product, not a personal one.
3. Fill in the form below.
4. Verify the site. Riot gives a code, which must be served as plain text in `riot.txt` at the root of the product URL's site; it is in the page's repository, [effe-exe/effe-exe.github.io](https://github.com/effe-exe/effe-exe.github.io). The portal adds `/riot.txt` to the URL as typed: for `https://effe-exe.github.io/` it asks for `https://effe-exe.github.io//riot.txt`, which GitHub Pages serves too.
5. Riot's Developer Relations team reviews it. The time it takes "can vary depending on your project and the application's target region(s)", and the answer comes as a message in the portal. Riot may grant, deny or revoke a key at its discretion.

## The form, as prepared

| Field | Answer |
|---|---|
| Product Name | Wardeye |
| Product Description | The text below. The box takes 1,500 characters |
| Product Group | Default Group |
| Product URL | `https://effe-exe.github.io`, Wardeye's own page ([D-027](decisions.md#d-027-wardeyes-own-page-at-effe-exegithubio)) |
| Product Game Focus | Riftbound |
| Are you organizing tournaments? | No |

Riot accepts a prototype or a detailed mock-up if it "clearly express[es] your product's purpose and the user flow". The page shows the flow; the demo video ([shot list](wardeye-page-brief.md#demo-video-shot-list-60-to-90-s)) goes on it once it is recorded.

## The description, as prepared (1,443 characters)

> Wardeye is a free, open-source Chrome and Edge extension for people watching Riftbound on Twitch, live or on replay. It recognises the face-up cards on the table in the video; pointing at one shows its name and image. A side panel lists each player's cards and the plays. It never records or re-hosts video.
>
> I'm Federico Vietti, an individual developer; this is a free community project. I also make Gradeon, a card pre-grading app, but Wardeye is not a Gradeon product: no ads, paid features, accounts or upsell.
>
> API: riftbound-content-v1 only. Card names, text (official English) and art are what a viewer sees on a card; the art also builds the recognition index, rebuilt each set. The key stays on my caching server, never in the extension.
>
> Flow: install from the Chrome Web Store, open a Riftbound stream or replay on Twitch, point at a card.
>
> Rules: for spectators, not a play aid (no rules enforcement, simulation, rankings or ratings). Public information only: never hand cams, face-down cards or hand lists. No metagame data: nothing is collected or published. Runs on the viewer's computer. No Riot logos; Legal Jibber Jabber notice shown.
>
> Status: alpha in Chrome Web Store review. Until approved it shows names and images from Riot's public card gallery; then only API data.
>
> Questions: does a spectator overlay fit? Its models were trained on card art and public broadcast frames: is that OK?
>
> Code: github.com/effe-exe/Wardeye

## A longer description, if Riot asks

> **Wardeye** is a free, open-source browser extension (Chrome and Edge, desktop) for people watching Riftbound on Twitch, live or on replay. While the video plays, it recognises the face-up cards on the table in the broadcast picture and outlines them. The viewer points at a card to see its name and card image. A side panel shows each player's face-up cards and the plays as they happen; on a replay, clicking a play jumps the video to it. It draws on top of the official Twitch player and never records, re-hosts or re-streams the video.
>
> **The maker.** Federico Vietti makes Wardeye as a free community project. He also makes Gradeon, a card pre-grading app, but Wardeye is not a Gradeon product: no ads, no paid features, no account, no upsell.
>
> **How it will use the API.**
> - The only API: `riftbound-content-v1`.
> - Card names, types and text, in official English, and the card art are what Wardeye shows when a viewer points at a card and in the side panel.
> - The same card art builds the recognition index: each card's art becomes a numeric fingerprint that the extension compares with the cards it sees on the table. It is rebuilt when a new set is released.
> - The key will stay on a small server that fetches the content when the content version changes and caches it; the extension gets card data from that server. The key is never in the extension's code, and viewers never call the Riot API: a few calls per content update in total.
>
> **User flow.** Install Wardeye from the Chrome Web Store (Chrome or Edge). Open a Riftbound stream or replay on Twitch. Wardeye finds the table camera and outlines the face-up cards as they are played. Point at a card to see its image and name, including cards lying under other cards. Optionally, open the side panel for each player's side and the plays, and paste a player's published decklist so Wardeye looks only among those cards.
>
> **What it never does.** It is for spectators, not a play aid: no rules enforcement, no gameplay simulation, no matchmaking, rankings or player ratings. It reads only the table camera, never hand cams, face-down cards or hand lists, and never guesses hidden cards. It keeps no metagame data: no play rates, win rates or statistics across matches; the plays list lives only in the browser tab and is deleted when the tab closes. Recognition runs on the viewer's computer, and no video or frames leave it. It is free for everyone, uses no Riot logos and carries the Legal Jibber Jabber notice.
>
> **Status.** An alpha is in the Chrome Web Store's review. Until a key is approved, it shows card names and images from Riot's public card gallery, loaded in the viewer's browser; it redistributes no card image or card text. Once a key is approved, all card data and art will come from the Riot API only, and anything Riot asks will be changed or withdrawn.

## The developer policies, point by point

Registering means agreeing to Riot's third-party developer policies, a "Please Don't" list shown before the form. The table maps each point to Wardeye.

| Riot's "Please Don't" | Wardeye |
|---|---|
| Break the law | — |
| Use any of our official logos | No Riot logos anywhere, in the product, repository or store listing |
| Refer to your project as a partnership with Riot, or as approved by Riot | Only the LJJ notice ("Riot Games does not endorse or sponsor this project"). Even with a key: never "approved by Riot", "official" or "partner" |
| Publish a project that doesn't properly secure your API key | The key lives only on our server. The extension never holds it |
| Use a Development or Interim key to run a project the community can access | Those keys only for the prototype Riot reviews. The public release waits for the production key |
| Use one production key for several projects | One key, one product. Any other project applies on its own |
| Compromise the integrity of the game, or give players an unfair advantage | Spectators only, public information only ([D-005](decisions.md)): it never reads hands, face-down cards or hand lists. It is not a play aid |
| Charge money, or give exclusive access to some users | Free for everyone. No paid features, and no account needed for any of it |
| Shame players, or give alternate channels to report or evaluate them | No player ratings, scores or reports. The timeline shows plays, not judgements |
| Build alternatives to official skill rankings | None. No cross-match statistics ([D-012](decisions.md#d-012-no-cross-match-statistics)) |
| Connect to other Riot systems (chat and so on) | None |
| Scrape undocumented endpoints or any source outside the Riot API and documented tools | **This changes things.** Once registered, card names, text and art come only from the Riot API. The path through the public card gallery (D-015) is retired. The recognition gallery and the models' card art are rebuilt from API assets. See App Notes |
| Build anything that looks like League of Legends' or Riot's own branding and design | Wardeye has its own brand book: an abstract ward mark (a stake and an orb, our own drawing), violet on near-black, Space Grotesk and Inter. Nothing of Riot's |

The "Please Do" list welcomes art assets from the game (never the logos): we use the card art to recognise cards.

## App Notes: the grey areas to ask Riot about

Riot invites questions about grey areas. The description asks the first three in short and answers the fourth; ask them in full if Riot writes back:

1. **The category.** An overlay for spectators of broadcasts and replays is not a deckbuilder or a card library, the approved examples. Does it fit, and on which terms?
2. **Recognition from the video itself.** Wardeye recognises the cards shown in the broadcast picture on the viewer's computer. Is the broadcast video an acceptable source next to the Riot API? The card names and art it shows would come from the API.
3. **The models.** The recognition models were trained on the official card art and on frames of public broadcasts. How does Riot want the art sourced (through the API, we assume), and may the broadcast frames be used for training?
4. **The maker.** Wardeye is Federico Vietti's community project. He also runs Gradeon, a company; the demo page is hosted on gradeon.ai, and the README credits Gradeon as the maker's other work. Wardeye is not a Gradeon product, has no ads, sales or upsell, and is free for everyone. Is that acceptable?

## What changes once a key is approved

- [D-015](decisions.md#d-015-no-riot-api-no-riot-assets-distributed) is superseded: card data and art come from the Riot API, and nothing else.
- The key stays on a small server that the extension asks for card data, as [D-011](decisions.md#d-011-card-data-and-art-come-only-from-the-riot-api) planned, because "Your API key may not be included in your code".
- Wardeye stays the Maintainer's community project, free for everyone, and never a Gradeon product ([D-024](decisions.md#d-024-wardeye-has-its-own-brand-book)). Never state or imply Riot's endorsement or approval. Keep the notice.
