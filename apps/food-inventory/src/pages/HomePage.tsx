import { Barcode, Minus, Plus } from "lucide-react";

interface HomePageProps {
  itemCount: number;
  onPurchase: () => void;
  onConsume: () => void;
  onManualAdd: () => void;
}

export function HomePage({ itemCount, onPurchase, onConsume, onManualAdd }: HomePageProps) {
  return (
    <section className="page home-page">
      <header className="page-header">
        <p className="eyebrow">Lokaler Bestand</p>
        <h1>Mein Vorrat</h1>
      </header>
      <div className="inventory-summary" aria-label="Bestandszusammenfassung">
        <strong>{itemCount}</strong>
        <span>Artikel im Bestand</span>
      </div>
      <div className="primary-actions">
        <button className="scan-action purchase" onClick={onPurchase} type="button">
          <Plus aria-hidden="true" size={28} />
          <span>Gekauft</span>
          <small>Manuell hinzufuegen</small>
        </button>
        <button className="scan-action consume" onClick={onConsume} type="button">
          <Minus aria-hidden="true" size={28} />
          <span>Verbraucht</span>
          <small>Im Bestand auswaehlen</small>
        </button>
      </div>
      <button className="manual-action" onClick={onManualAdd} type="button">
        <Barcode aria-hidden="true" size={20} />
        Ohne Barcode hinzufuegen
      </button>
      <p className="muted-status">Barcode-Scan und Synchronisierung folgen in einer spaeteren Version.</p>
    </section>
  );
}
