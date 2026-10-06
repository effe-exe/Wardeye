// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The few extension APIs Wardeye uses, typed here so the repository needs no @types/chrome.
//
// Messages arrive typed `any`: they are JSON from another part of the extension, and each listener names the shape it
// reads (and checks its `kind`) rather than narrowing an `unknown` first.

declare namespace chrome.runtime {
  interface MessageSender {
    tab?: { id?: number };
  }
  interface Port {
    name: string;
    sender?: MessageSender;
    postMessage(message: unknown): void;
    disconnect(): void;
    onMessage: { addListener(cb: (message: any, port: Port) => void): void };
    onDisconnect: { addListener(cb: (port: Port) => void): void };
  }
  function connect(info?: { name?: string }): Port;
  const onConnect: { addListener(cb: (port: Port) => void): void };
  function getURL(path: string): string;
  function sendMessage(message: unknown): Promise<any>;
  const onMessage: {
    addListener(cb: (message: any, sender: MessageSender, sendResponse: (response?: unknown) => void) => boolean | void): void;
  };
  function getContexts(filter: { contextTypes: string[] }): Promise<{ contextType: string }[]>;
}

declare namespace chrome.offscreen {
  function createDocument(parameters: { url: string; reasons: string[]; justification: string }): Promise<void>;
  function closeDocument(): Promise<void>;
}

declare namespace chrome.action {
  function setBadgeText(details: { tabId?: number; text: string }): Promise<void>;
  function setBadgeBackgroundColor(details: { tabId?: number; color: string }): Promise<void>;
  function setTitle(details: { tabId?: number; title: string }): Promise<void>;
  function getBadgeText(details: { tabId?: number }): Promise<string>;
  function getTitle(details: { tabId?: number }): Promise<string>;
  // onClicked, (tab) => void, is declared once for both apps in apps/bench/src/chrome.d.ts: the namespaces merge
}

declare namespace chrome.tabs {
  function sendMessage(tabId: number, message: unknown): Promise<any>;
  /** The plays panel's tab (and, in the tests, the worker's own view of the tabs): no "tabs" permission, so no addresses. */
  function query(queryInfo: { active?: boolean; currentWindow?: boolean }): Promise<{ id?: number }[]>;
  const onActivated: { addListener(cb: (info: { tabId: number; windowId: number }) => void): void };
  const onUpdated: { addListener(cb: (tabId: number, change: { status?: string }) => void): void };
  /** The plays panel's line to a tab's content script (its runtime.onConnect). */
  function connect(tabId: number, info?: { name?: string }): chrome.runtime.Port;
}

declare namespace chrome.sidePanel {
  /** Chrome 116 and later, in answer to a click (the badge's plays button, forwarded by the content script). */
  function open(options: { tabId?: number; windowId?: number }): Promise<void>;
}

declare namespace chrome.storage {
  /** The viewer's settings (the plays panel's performance level and view): the only things Wardeye keeps. */
  const local: { get(keys: string | string[]): Promise<Record<string, unknown>>; set(items: Record<string, unknown>): Promise<void> };
  const onChanged: { addListener(cb: (changes: Record<string, { newValue?: unknown }>, area: string) => void): void };
}
