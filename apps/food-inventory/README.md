# Mein Vorrat

Eine local-first PWA zum Verwalten von Lebensmitteln. Bestandsaenderungen werden
als unveraenderliche Ereignisse in IndexedDB gespeichert. Sie bleibt mit bekannten
Produkten offline nutzbar und kann optional einen gemeinsamen Supabase-Haushalt
automatisch synchronisieren.

## Entwicklungsstand

MVP 1 ist umgesetzt: React/TypeScript/Vite, PWA-Grundlage, Dexie-Schema,
manuelle Produkterfassung, append-only Bestandsereignisse, `+1`/`-1`,
Produktbearbeitung, lokaler Barcode-Scan, mobile Navigation und optionaler
Haushalts-Sync. Bekannte Barcodes funktionieren nach ihrer ersten Zuordnung auch
ohne Netzwerk.

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
anschliessend "Zum Home-Bildschirm" auswaehlen.

## Gemeinsamer Sync mit Supabase

Der Sync ist absichtlich optional. Ohne Konfiguration bleibt die App lokal.

1. Bei [Supabase](https://supabase.com/) ein kostenloses Projekt anlegen.
2. Den Inhalt von `supabase/schema.sql` im **SQL Editor** des Projekts ausfuehren.
3. Unter **Authentication > Sign In / Providers** **Anonymous Sign-Ins** aktivieren.
   Es werden weder E-Mail-Adressen noch Passwoerter verwendet.
4. Unter **Project Settings > API** die Project URL und den **Publishable** Key
   unter GitHub **Settings > Secrets and variables > Actions > Variables** als
   `VITE_SUPABASE_URL` und `VITE_SUPABASE_ANON_KEY` hinterlegen. Diese Werte sind fuer Browser-Apps
   bestimmt; der `service_role` Key darf nie verwendet oder gespeichert werden.
5. Den Deploy-Workflow erneut ausfuehren. Danach erscheint der Tab **Gemeinsam**:
   Auf beiden Geraeten zuerst **Dieses Geraet verbinden** waehlen. Auf dem ersten
   Geraet einen Haushalt erstellen, auf dem zweiten den angezeigten Einladungscode
   eingeben.

Wurde das Schema bereits vor dem 12. September 2026 angelegt, den Inhalt von
`supabase/migrations/20260912_grant_sync_access.sql` einmal zusaetzlich im SQL
Editor ausfuehren. Die Rechte machen nur die Data API erreichbar; RLS bleibt fuer
die Haushaltsgrenzen verantwortlich.

Die anonyme Geraeteidentitaet liegt nur im Browser. Werden Browserdaten geloescht
oder die App auf einem neuen Geraet installiert, muss dieses Geraet erneut ueber
den Einladungscode mit dem Haushalt verbunden werden.

Beim Start, nach lokalen Aenderungen, nach einer Netzwerkrueckkehr und bei einer
Realtime-Aenderung des gemeinsamen Haushalts gleicht die App Ereignisse ab.
Ereignisse werden per UUID zusammengefuehrt; sie ueberschreiben sich nicht.

## GitHub Pages

Der Deployment-Workflow veroeffentlicht die App unter
`https://robfil.github.io/food-inventory/`. In den Repository-Einstellungen unter
**Pages** muss einmalig **GitHub Actions** als Publishing Source aktiviert werden.
Der Workflow baut dabei die bestehende Website und die App gemeinsam.

## Noch nicht enthalten

Noch keine Open-Food-Facts-Abfrage und kein GitHub-Sync. Ein unbekannter Barcode
wird beim Kauf derzeit einmalig manuell benannt; beim Verbrauch wird er nicht
gebucht. Es werden weder Zugangsdaten noch Tokens im Projekt gespeichert.
