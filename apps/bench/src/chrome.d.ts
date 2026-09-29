// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The few extension APIs the bench uses, typed here so the repository needs no @types/chrome. (The extension
// declares its own in apps/extension/src/chrome.d.ts; the namespaces merge, the members differ.)

declare namespace chrome.runtime {
  const onInstalled: { addListener(cb: () => void): void };
  function getURL(path: string): string;
}

declare namespace chrome.action {
  // shared with the extension, whose toolbar button reads the tab it was clicked in (one declaration for both: they merge)
  const onClicked: { addListener(cb: (tab: { id?: number }) => void): void };
}

declare namespace chrome.tabs {
  function create(properties: { url: string }): Promise<unknown>;
}
