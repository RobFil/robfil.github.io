# Mein Vorrat

Eine local-first PWA zum Verwalten von Lebensmitteln. Bestandsaenderungen werden
als unveraenderliche Ereignisse in IndexedDB gespeichert. Die App benoetigt keinen
eigenen Server und bleibt mit bekannten Produkten offline nutzbar.

## Entwicklungsstand

MVP 1 ist umgesetzt: React/TypeScript/Vite, PWA-Grundlage, Dexie-Schema,
manuelle Produkterfassung, append-only Bestandsereignisse, `+1`/`-1`,
Produktbearbeitung, lokaler Barcode-Scan und mobile Navigation. Bekannte
Barcodes funktionieren nach ihrer ersten Zuordnung auch ohne Netzwerk.

## Installation und Start

Node.js 20 oder neuer installieren, dann im Ordner dieser App ausfuehren:

```powershell
npm install
npm run dev
```

## Qualitaetspruefungen

```powershell
npm run lint
npm run test
npm run build
```

## PWA und iPhone

Im Entwicklungsmodus funktioniert die App im Browser. Fuer Kamera und Installation
auf dem iPhone muss sie ueber HTTPS bereitgestellt werden. In Safari: Teilen und
anschliessend "Zum Home-Bildschirm" auswaehlen. Die Kamera wird erst mit MVP 2
eingebunden.

## GitHub Pages

Der Deployment-Workflow veroeffentlicht die App unter
`https://robfil.github.io/food-inventory/`. In den Repository-Einstellungen unter
**Pages** muss einmalig **GitHub Actions** als Publishing Source aktiviert werden.
Der Workflow baut dabei die bestehende Website und die App gemeinsam.

## Grenzen von MVP 0

Noch keine Open-Food-Facts-Abfrage und kein GitHub-Sync. Ein unbekannter Barcode
wird beim Kauf derzeit einmalig manuell benannt; beim Verbrauch wird er nicht
gebucht. Es werden weder Zugangsdaten noch Tokens im Projekt gespeichert.
