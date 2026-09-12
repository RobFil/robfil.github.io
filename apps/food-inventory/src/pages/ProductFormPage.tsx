import { ArrowLeft, Save } from "lucide-react";
import { type FormEvent, useState } from "react";
import type { Product } from "../domain/types";

interface ProductFormPageProps {
  product?: Product;
  onCancel: () => void;
  onSave: (input: { name: string; genericIngredient: string | null }, addToStock: boolean) => Promise<void>;
}

export function ProductFormPage({ product, onCancel, onSave }: ProductFormPageProps) {
  const [name, setName] = useState(product?.name ?? "");
  const [ingredient, setIngredient] = useState(product?.genericIngredient ?? "");
  const [addToStock, setAddToStock] = useState(!product);
  const [saving, setSaving] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!name.trim()) return;
    setSaving(true);
    await onSave({ name, genericIngredient: ingredient || null }, addToStock);
    setSaving(false);
  }

  return (
    <section className="page form-page">
      <header className="form-header">
        <button aria-label="Zurueck" className="icon-button" onClick={onCancel} type="button">
          <ArrowLeft aria-hidden="true" size={22} />
        </button>
        <h1>{product ? "Produkt bearbeiten" : "Produkt hinzufuegen"}</h1>
      </header>
      <form className="product-form" onSubmit={handleSubmit}>
        <label>
          Produktname
          <input autoComplete="off" autoFocus onChange={(event) => setName(event.target.value)} required value={name} />
        </label>
        <label>
          Zutat (optional)
          <input autoComplete="off" onChange={(event) => setIngredient(event.target.value)} placeholder="z. B. Milch" value={ingredient} />
        </label>
        {!product && (
          <label className="checkbox-field">
            <input checked={addToStock} onChange={(event) => setAddToStock(event.target.checked)} type="checkbox" />
            Gleich zum Bestand hinzufuegen (+1)
          </label>
        )}
        <button className="submit-button" disabled={saving || !name.trim()} type="submit">
          <Save aria-hidden="true" size={20} />
          {saving ? "Wird gespeichert" : "Speichern"}
        </button>
      </form>
    </section>
  );
}
