# Architektur

## Local first

Die Anwendung arbeitet vollstaendig lokal. IndexedDB ist die Datenquelle auf dem
Geraet; Netzwerkzugriffe sind nur fuer unbekannte Barcodes und einen spaeteren,
optionalen GitHub-Sync vorgesehen.

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

## Spaeterer GitHub-Sync

`SyncAdapter` ist bereits als Vertrag vorhanden, aber nicht implementiert. Ein
`GitHubSyncAdapter` soll Produkte mit Last-Write-Wins und Ereignisse per UUID-Menge
zusammenfuehren. Zugangstokens bleiben ausschliesslich lokal auf dem Geraet.
