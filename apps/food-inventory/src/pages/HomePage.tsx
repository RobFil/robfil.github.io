import { Barcode, Minus, Plus } from "lucide-react";

export function HomePage() {
  return (
    <section className="page home-page">
      <header className="page-header">
        <p className="eyebrow">Lokaler Bestand</p>
        <h1>Mein Vorrat</h1>
      </header>
      <div className="inventory-summary" aria-label="Bestandszusammenfassung">
        <strong>0</strong>
        <span>Artikel im Bestand</span>
      </div>
      <div className="primary-actions">
        <button className="scan-action purchase" type="button">
          <Plus aria-hidden="true" size={28} />
          <span>Gekauft</span>
          <small>Barcode scannen</small>
        </button>
        <button className="scan-action consume" type="button">
          <Minus aria-hidden="true" size={28} />
          <span>Verbraucht</span>
          <small>Barcode scannen</small>
        </button>
      </div>
      <button className="manual-action" type="button">
        <Barcode aria-hidden="true" size={20} />
        Ohne Barcode hinzufuegen
      </button>
      <p className="muted-status">Synchronisierung wird in einer spaeteren Version verfuegbar sein.</p>
    </section>
  );
}
