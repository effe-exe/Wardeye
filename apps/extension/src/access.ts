// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Where Wardeye runs beyond Twitch (decision D-037). Twitch is in the manifest. YouTube is an optional permission, for two
// reasons: an update that adds a site to the manifest switches the extension off for everyone until they accept it, and
// nobody who never watches Riftbound on YouTube has to give Wardeye access to it. The viewer allows it once, with a click on
// Wardeye's toolbar button on a YouTube page. From then on the overlay's script is registered for youtube.com and runs on
// its pages as it does on Twitch's. Taking the permission back (in the browser's extension settings) unregisters it.

/** The pages of YouTube Wardeye asks for (the manifest's optional_host_permissions; a test holds them together). */
export const YOUTUBE_ORIGINS: readonly string[] = ['https://www.youtube.com/*'];
/** The overlay's script, registered for YouTube under this id. */
export const SCRIPT_ID = 'wardeye-youtube';

/** What this file needs of the browser: chrome's APIs in the extension (chromeBrowser), a fake in the tests. */
export interface Browser {
  /** Whether these origins are allowed; asking shows the browser's prompt, the first time only. */
  allowed(origins: readonly string[]): Promise<boolean>;
  ask(origins: readonly string[]): Promise<boolean>;
  /** The ids of the content scripts registered now. */
  registered(): Promise<string[]>;
  register(id: string, origins: readonly string[]): Promise<void>;
  unregister(id: string): Promise<void>;
  /** Whether the overlay's script already runs in the tab. */
  present(tabId: number): Promise<boolean>;
  /** Runs the overlay's script in a tab whose page was open before it was allowed. */
  inject(tabId: number): Promise<void>;
  /** Turns Wardeye off or on in the tab (what the toolbar button does on Twitch). */
  toggle(tabId: number): Promise<void>;
}

/** The registered script follows the permission: registered while YouTube is allowed, gone once it is not. */
export async function sync(b: Browser): Promise<void> {
  const allowed = await b.allowed(YOUTUBE_ORIGINS);
  const there = (await b.registered()).includes(SCRIPT_ID);
  if (allowed && !there) await b.register(SCRIPT_ID, YOUTUBE_ORIGINS);
  if (!allowed && there) await b.unregister(SCRIPT_ID);
}

/** A page of YouTube (the toolbar button's tab; its address is known for that click only, from the activeTab permission). */
export function onYouTube(url: string | undefined): boolean {
  try {
    return new URL(url ?? '').hostname === 'www.youtube.com';
  } catch {
    return false;
  }
}

/** The toolbar button, clicked on a tab. On Twitch, and on YouTube once Wardeye runs there, it turns Wardeye off or on.
 * On YouTube before that, it asks for YouTube, then starts Wardeye in the tab at once, with no reload. */
export async function clicked(b: Browser, tab: { id?: number; url?: string }): Promise<void> {
  if (tab.id === undefined) return;
  if (!onYouTube(tab.url)) return b.toggle(tab.id);
  // Asked first, before anything is awaited: the browser lets an extension ask only while it answers the click. Once YouTube
  // is allowed, asking again shows nothing and says yes.
  const asked = b.ask(YOUTUBE_ORIGINS);
  if (!(await asked)) return;
  await sync(b);
  if (await b.present(tab.id)) return b.toggle(tab.id);
  await b.inject(tab.id);
}

/** The Browser of the extension's worker: chrome's permissions, scripting and tabs APIs. */
export function chromeBrowser(): Browser {
  return {
    allowed: (origins) => chrome.permissions.contains({ origins: [...origins] }),
    ask: (origins) => chrome.permissions.request({ origins: [...origins] }),
    registered: async () => (await chrome.scripting.getRegisteredContentScripts()).map((s) => s.id),
    register: (id, origins) =>
      chrome.scripting.registerContentScripts([{ id, matches: [...origins], js: ['content.js'], css: ['overlay.css'], runAt: 'document_idle', persistAcrossSessions: true }]),
    unregister: (id) => chrome.scripting.unregisterContentScripts({ ids: [id] }),
    present: (tabId) => chrome.tabs.sendMessage(tabId, { kind: 'hello' }).then(() => true, () => false),
    inject: async (tabId) => {
      await chrome.scripting.insertCSS({ target: { tabId }, files: ['overlay.css'] });
      await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] });
    },
    toggle: (tabId) => chrome.tabs.sendMessage(tabId, { kind: 'toggle' }).then(() => {}, () => {}), // a page Wardeye does not read
  };
}
