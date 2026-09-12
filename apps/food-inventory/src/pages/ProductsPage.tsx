import { Plus } from "lucide-react";
import type { InventoryItem } from "../domain/inventory";

interface ProductsPageProps {
  items: InventoryItem[];
  onAdd: () => void;
  onEdit: (productId: string) => void;
}

export function ProductsPage({ items, onAdd, onEdit }: ProductsPageProps) {
  return (
    <section className="page products-page">
      <header className="page-header page-header-row"><h1>Produkte</h1><button aria-label="Produkt hinzufuegen" className="icon-button" onClick={onAdd} type="button"><Plus aria-hidden="true" size={22} /></button></header>
      {items.length === 0 ? <div className="empty-state"><p>Lege dein erstes Produkt an.</p></div> : (
        <div className="product-list">
          {items.map((item) => <button className="product-list-item" key={item.product.id} onClick={() => onEdit(item.product.id)} type="button"><span><strong>{item.product.name}</strong><small>{item.product.genericIngredient ?? "Keine Zutat hinterlegt"}</small></span><b>{item.quantity}</b></button>)}
        </div>
      )}
    </section>
  );
}
