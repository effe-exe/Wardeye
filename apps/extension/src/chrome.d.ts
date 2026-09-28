// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The few extension APIs RiftEye uses, typed here so the repository needs no @types/chrome.

declare namespace chrome.runtime {
  interface Port {
    name: string;
    postMessage(message: unknown): void;
    disconnect(): void;
    onMessage: { addListener(cb: (message: any, port: Port) => void): void };
    onDisconnect: { addListener(cb: (port: Port) => void): void };
  }
  function connect(info?: { name?: string }): Port;
  const onConnect: { addListener(cb: (port: Port) => void): void };
}
