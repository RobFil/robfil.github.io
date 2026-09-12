import { Minus, PackagePlus, Plus } from "lucide-react";
import type { InventoryItem } from "../domain/inventory";

interface InventoryPageProps {
  items: InventoryItem[];
  onAdd: (productId: string) => Promise<void>;
  onConsume: (productId: string) => Promise<void>;
  onEdit: (productId: string) => void;
}

export function InventoryPage({ items, onAdd, onConsume, onEdit }: InventoryPageProps) {
  const stockedItems = items.filter((item) => item.quantity > 0);

  return (
    <section className="page inventory-page">
      <header className="page-header"><h1>Bestand</h1></header>
      {stockedItems.length === 0 ? (
        <div className="empty-state"><PackagePlus aria-hidden="true" size={34} /><p>Dein Bestand ist noch leer.</p></div>
      ) : (
        <div className="inventory-list">
          {stockedItems.map((item) => (
            <article className="inventory-row" key={item.product.id}>
              <button className="product-summary" onClick={() => onEdit(item.product.id)} type="button">
                <strong>{item.product.name}</strong>
                {item.product.genericIngredient && <span>{item.product.genericIngredient}</span>}
              </button>
              <div className="quantity-controls">
                <button aria-label={`${item.product.name} verbrauchen`} className="quantity-button" onClick={() => void onConsume(item.product.id)} type="button"><Minus aria-hidden="true" size={18} /></button>
                <output aria-label={`${item.quantity} im Bestand`}>{item.quantity}</output>
                <button aria-label={`${item.product.name} hinzufuegen`} className="quantity-button" onClick={() => void onAdd(item.product.id)} type="button"><Plus aria-hidden="true" size={18} /></button>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
