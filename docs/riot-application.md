# Applying to Riot for a Riftbound app key

**Status: draft, not sent.** Why apply: Riot's fan-content policy (the Legal Jibber Jabber, LJJ) counts "any Project that involves a business or legal entity" as commercial, even a free one. So the extension can be published as a tool from Gradeon only once it has one of these:
- a written licence from Riot;
- an approved Riftbound app key (LJJ §2's carve-out for "commercial Projects that both (1) comply with our API Terms and API Policies; and (2) use a currently valid Riot API key").

Until then it is a free, non-commercial community project ([D-015](decisions.md#d-015-no-riot-api-no-riot-assets-distributed), [08 §8.2](research/08-legal-and-policy.md#82-riot-games-policies)). This is a risk analysis, not legal advice. Re-read the live pages before sending: [the Riftbound page](https://developer.riotgames.com/docs/riftbound), [the Riftbound policy](https://developer.riotgames.com/policies/riftbound), [product registration](https://developer.riotgames.com/docs/portal) and [the LJJ](https://www.riotgames.com/en/legal).

## Steps

1. Sign in at [developer.riotgames.com](https://developer.riotgames.com) with the Riot account that will own the product. That is Gradeon's, if Gradeon is to publish it.
2. Click **Register Product** and choose the larger-scale (production) product, not a personal project.
3. Choose Riftbound and fill in the form below. When the portal asks you to verify the product, follow its instructions.
4. Riot's Developer Relations team reviews it. The time it takes "can vary depending on your project and the application's target region(s)". Riot may grant, deny or revoke a key at its discretion.

## What the form asks for

| Field | Answer |
|---|---|
| Name and contact | The applicant (Gradeon or Federico Vietti) and a contact address |
| Description: use case, a demonstration, the user flow | The draft below, and a 60 to 90 s screen recording |
| Distribution location | Chrome Web Store; the source on GitHub |
| Target platforms | Desktop Chrome and Edge; Firefox later |
| A valid URL where the app can be accessed or downloaded | **To prepare:** a public page with the demo video and the download (a landing page, or the public repository) |

Riot accepts a prototype or a detailed mock-up if it "clearly express[es] your product's purpose and the user flow". The standalone alpha and a recording of it on a broadcast are enough.

## Draft description

> **Wardeye** is a free, open-source browser extension for people watching Riftbound on Twitch. It recognises the cards on the table in the broadcast video, so a viewer can point at a card and see its name and official image. It adds a spectator's reference layer next to the official player; it never re-hosts the video.
>
> **For spectators, not players.** It works on broadcasts and replays. It is not a play aid: no automated rules enforcement, no gameplay simulation, no matchmaking or rankings, and no metagame statistics (no play or win rates).
>
> **Public information only.** It reads only the table camera. It never reads hand cams, face-down cards or the broadcast's hand lists.
>
> **Free and private.** Everything is free. The recognition runs on the viewer's computer, and no video leaves it.
>
> **Card data.** With a key, card names, text and images come from the Riot API, as the policy requires. The key stays on a small server of ours and never ships in the extension.
>
> **User flow.**
> 1. Install the extension.
> 2. Open a Riftbound stream or replay on Twitch.
> 3. Boxes appear on the cards on the table.
> 4. Point at one to see the card.
>
> It carries the Legal Jibber Jabber notice and uses no Riot logos.

## The developer policies, point by point

Registering means agreeing to Riot's third-party developer policies, a "Please Don't" list shown before the form. The table maps each point to Wardeye.

| Riot's "Please Don't" | Wardeye |
|---|---|
| Break the law | — |
| Use any of our official logos | No Riot logos anywhere, in the product, repository or store listing |
| Refer to your project as a partnership with Riot, or as approved by Riot | Only the LJJ notice ("Riot Games does not endorse or sponsor this project"). Even with a key: never "approved by Riot", "official" or "partner" |
| Publish a project that doesn't properly secure your API key | The key lives only on our server. The extension never holds it |
| Use a Development or Interim key to run a project the community can access | Those keys only for the prototype Riot reviews. The public release waits for the production key |
| Use one production key for several projects | One key, one product. Every further Gradeon tool applies on its own |
| Compromise the integrity of the game, or give players an unfair advantage | Spectators only, public information only ([D-005](decisions.md)): it never reads hands, face-down cards or hand lists. It is not a play aid |
| Charge money, or give exclusive access to some users | Free for everyone. No paid features, and no Gradeon account needed for any of it |
| Shame players, or give alternate channels to report or evaluate them | No player ratings, scores or reports. The timeline shows plays, not judgements |
| Build alternatives to official skill rankings | None. No cross-match statistics ([D-012](decisions.md#d-012-no-cross-match-statistics)) |
| Connect to other Riot systems (chat and so on) | None |
| Scrape undocumented endpoints or any source outside the Riot API and documented tools | **This changes things.** Once registered, card names, text and art come only from the Riot API. The path through the public card gallery (D-015) is retired. The recognition gallery and the models' card art are rebuilt from API assets. See App Notes |
| Build anything that looks like League of Legends' or Riot's own branding and design | Wardeye wears Gradeon's look (near-black, purple, Space Mono), nothing of Riot's |

The "Please Do" list welcomes art assets from the game (never the logos): we use the card art to recognise cards.

## App Notes: the grey areas to ask Riot about

Riot invites questions about grey areas as App Notes in the application. Ask:

1. **The category.** An overlay for spectators of broadcasts and replays is not a deckbuilder or a card library, the approved examples. Does it fit, and on which terms?
2. **Recognition from the video itself.** Wardeye recognises the cards shown in the broadcast picture on the viewer's computer. Is the broadcast video an acceptable source next to the Riot API? The card names and art it shows would come from the API.
3. **The models.** The recognition models were trained on the official card art and on frames of public broadcasts. How does Riot want the art sourced (through the API, we assume), and may the broadcast frames be used for training?
4. **Gradeon as the developer.** Gradeon, a company, builds and credits Wardeye. It is free for everyone, with no ads, sales or upsell, and a plain "by Gradeon" credit and link. Is that acceptable?

## What changes once a key is approved

- [D-015](decisions.md#d-015-no-riot-api-no-riot-assets-distributed) is superseded: card data and art come from the Riot API, and nothing else.
- The key stays on a small server that the extension asks for card data, as [D-011](decisions.md#d-011-card-data-and-art-come-only-from-the-riot-api) planned, because "Your API key may not be included in your code".
- The extension, README and store listing may then say "by Gradeon". Keep them free for everyone. Never state or imply Riot's endorsement or approval. Keep the notice.
