import { defineConfig } from '@playwright/test';

// Browser end-to-end tests for the apps. CI installs Playwright's Chromium; locally you can
// point at any Chromium build with RIFTEYE_CHROMIUM=/path/to/chrome.
export default defineConfig({
  testDir: 'apps',
  testMatch: '**/e2e/*.spec.ts',
  timeout: 60_000,
  reporter: [['list']],
  use: {
    headless: true,
    ...(process.env.RIFTEYE_CHROMIUM ? { launchOptions: { executablePath: process.env.RIFTEYE_CHROMIUM } } : {}),
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
});
