// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The extension's background worker: opens the bench page in a tab when the extension is installed (or
// reloaded) and whenever its button is clicked. Nothing else runs here; the bench itself runs in the page.

const openBench = (): void => {
  void chrome.tabs.create({ url: chrome.runtime.getURL('bench.html') });
};

chrome.runtime.onInstalled.addListener(openBench);
chrome.action.onClicked.addListener(openBench);

export {}; // a module, so that its names stay out of the global scope the other apps' scripts share
