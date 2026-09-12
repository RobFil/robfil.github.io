import { ArrowLeft, Save } from "lucide-react";
import { type FormEvent, useState } from "react";
import type { Product, StorageLocation } from "../domain/types";

interface ProductFormPageProps {
  product?: Product;
  onCancel: () => void;
  onSave: (input: { name: string; genericIngredient: string | null; storageLocation: StorageLocation }, addToStock: boolean) => Promise<void>;
}

const locations: Array<{ value: StorageLocation; label: string }> = [
  { value: "fridge", label: "Kuehlschrank" },
  { value: "freezer", label: "Gefrierfach" },
  { value: "pantry", label: "Vorratsschrank" },
  { value: "other", label: "Sonstiges" },
];

export function ProductFormPage({ product, onCancel, onSave }: ProductFormPageProps) {
  const [name, setName] = useState(product?.name ?? "");
  const [ingredient, setIngredient] = useState(product?.genericIngredient ?? "");
  const [location, setLocation] = useState<StorageLocation>(product?.storageLocation ?? "fridge");
  const [addToStock, setAddToStock] = useState(!product);
  const [saving, setSaving] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!name.trim()) return;
    setSaving(true);
    await onSave({ name, genericIngredient: ingredient || null, storageLocation: location }, addToStock);
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
          Zutat fuer spaeter
          <input autoComplete="off" onChange={(event) => setIngredient(event.target.value)} placeholder="z. B. Milch" value={ingredient} />
        </label>
        <label>
          Aufbewahrung
          <select onChange={(event) => setLocation(event.target.value as StorageLocation)} value={location}>
            {locations.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
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
