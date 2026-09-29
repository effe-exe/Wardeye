// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The few extension APIs Wardeye uses, typed here so the repository needs no @types/chrome.

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
  /** The tests only: the worker's own view of the tabs (no "tabs" permission, so no addresses). */
  function query(queryInfo: { active?: boolean }): Promise<{ id?: number }[]>;
}
