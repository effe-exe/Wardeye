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
