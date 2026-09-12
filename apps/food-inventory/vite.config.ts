import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      manifest: {
        name: "Mein Vorrat",
        short_name: "Vorrat",
        display: "standalone",
        orientation: "portrait",
        start_url: "/",
        background_color: "#f7faf8",
        theme_color: "#156f5c",
        icons: [{ src: "/icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" }],
      },
      workbox: {
        navigateFallback: "/index.html",
      },
    }),
  ],
});
