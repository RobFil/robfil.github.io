# Mein Vorrat

Eine local-first PWA zum Verwalten von Lebensmitteln. Bestandsaenderungen werden
als unveraenderliche Ereignisse in IndexedDB gespeichert. Die App benoetigt keinen
eigenen Server und bleibt mit bekannten Produkten offline nutzbar.

## Entwicklungsstand

MVP 0 ist angelegt: React/TypeScript/Vite, PWA-Grundlage, Dexie-Schema,
Domaintypen, Repository-Vertraege, GitHub-Sync-Stummel und mobile Navigation.
Barcode-Scanning und die eigentliche Bestandsverwaltung folgen in den naechsten
Milestones.

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

Noch keine Barcode-Scans, keine Open-Food-Facts-Abfrage und kein GitHub-Sync. Es
werden weder Zugangsdaten noch Tokens im Projekt gespeichert.
