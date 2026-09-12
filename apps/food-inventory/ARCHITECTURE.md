# Architektur

## Local first

IndexedDB ist die lokale Datenquelle auf dem Geraet. Dadurch funktionieren
bekannte Produkte, Scans und Bestandsaenderungen offline. Optional synchronisiert
ein Supabase-Adapter die lokalen Daten mit einem gemeinsamen Haushalt.

Ein Produkt besteht bewusst nur aus einem Namen, einer optionalen allgemeinen
Zutat und einer optionalen Barcode-Zuordnung. Details wie Aufbewahrungsort oder
Packungsgroesse werden erst ergaenzt, wenn sie einen konkreten Nutzen haben.

## Ereignisbasierter Bestand

`InventoryEvent` ist append-only. Der aktuelle Bestand eines Produkts ist die
Summe aller zugehoerigen `quantityChange`-Werte. Damit koennen Ereignisse mehrerer
Geraete spaeter anhand ihrer UUID zusammengefuehrt werden, ohne Bestandswerte zu
ueberschreiben.

## Schichten

- `src/domain`: reine Modelle und Geschaeftslogik
- `src/db`: Dexie-Datenbank und Repository-Vertraege
- `src/services`: externe Adapter, etwa Scanner, Open Food Facts und Sync
- `src/components` und `src/pages`: mobile React-Oberflaeche

## Gemeinsamer Supabase-Sync

Jede Person meldet sich mit einem eigenen Magic-Link an und tritt einem Haushalt
bei. Die Datenbank setzt Row Level Security durch: Ein Nutzer kann nur die Daten
von Haushalten lesen oder schreiben, deren Mitglied er ist. Ein zufaelliger
Einladungscode verbindet das zweite Geraet mit dem Haushalt.

Der Adapter zieht zuerst Produkte und Ereignisse. Ereignisse werden anhand ihrer
UUID vereinigt. Produkte verwenden bei Barcodes die stabile ID `barcode:<code>`;
dadurch entstehen bei zwei offline verwendeten Geraeten keine separaten
Produktidentitaeten. Namen ohne Barcode behalten eine UUID. Danach werden lokale
Produkte und noch nicht hochgeladene Ereignisse in die Cloud geschrieben.

Supabase Realtime benachrichtigt offene Anwendungen ueber Aenderungen. Zusaetzlich
versucht die App den Abgleich beim Start, nach einer lokalen Aenderung und sobald
eine Netzwerkverbindung zurueckkehrt. Fehler lassen die lokale Warteschlange
unveraendert, damit der naechste Versuch nichts verliert.

## Spaeterer GitHub-Sync

`SyncAdapter` ist weiterhin als Vertrag vorhanden, aber nicht implementiert. Ein
`GitHubSyncAdapter` soll Produkte mit Last-Write-Wins und Ereignisse per UUID-Menge
zusammenfuehren. Zugangstokens bleiben ausschliesslich lokal auf dem Geraet.
