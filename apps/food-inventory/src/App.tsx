import { useCallback, useEffect, useMemo, useState } from "react";
import { BottomNavigation, type PageId } from "./components/BottomNavigation";
import type { InventoryItem } from "./domain/inventory";
import type { Product } from "./domain/types";
import { HomePage } from "./pages/HomePage";
import { InventoryPage } from "./pages/InventoryPage";
import { ProductFormPage } from "./pages/ProductFormPage";
import { ProductsPage } from "./pages/ProductsPage";
import { consumeProduct, createProduct, getAllInventory, purchaseProduct, updateProduct } from "./services/inventoryService";

export default function App() {
  const [activePage, setActivePage] = useState<PageId>("home");
  const [items, setItems] = useState<InventoryItem[]>([]);
  const [formProduct, setFormProduct] = useState<Product | null>(null);
  const [formOpen, setFormOpen] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const refresh = useCallback(async () => setItems(await getAllInventory()), []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const stockedItemCount = useMemo(() => items.filter((item) => item.quantity > 0).length, [items]);

  function showNotice(message: string) {
    setNotice(message);
    window.setTimeout(() => setNotice(null), 2400);
  }

  function openNewProduct() {
    setFormProduct(null);
    setFormOpen(true);
  }

  function openProduct(productId: string) {
    const item = items.find((entry) => entry.product.id === productId);
    if (!item) return;
    setFormProduct(item.product);
    setFormOpen(true);
  }

  async function addOne(productId: string) {
    await purchaseProduct(productId);
    await refresh();
    showNotice("Zum Bestand hinzugefuegt.");
  }

  async function removeOne(productId: string) {
    if (!(await consumeProduct(productId))) {
      showNotice("Dieser Artikel ist nicht mehr im Bestand.");
      return;
    }
    await refresh();
    showNotice("Als verbraucht markiert.");
  }

  async function saveProduct(
    input: { name: string; genericIngredient: string | null },
    addToStock: boolean,
  ) {
    if (formProduct) {
      await updateProduct(formProduct, input);
      showNotice("Produkt gespeichert.");
    } else {
      await createProduct(input, addToStock);
      showNotice(addToStock ? "Produkt zum Bestand hinzugefuegt." : "Produkt gespeichert.");
    }
    await refresh();
    setFormOpen(false);
  }

  if (formOpen) {
    return (
      <main className="app-shell">
        <div className="app-content"><ProductFormPage onCancel={() => setFormOpen(false)} onSave={saveProduct} product={formProduct ?? undefined} /></div>
        {notice && <div className="toast" role="status">{notice}</div>}
      </main>
    );
  }

  const content = activePage === "home"
    ? <HomePage itemCount={stockedItemCount} onConsume={() => setActivePage("inventory")} onPurchase={openNewProduct} />
    : activePage === "inventory"
      ? <InventoryPage items={items} onAdd={addOne} onConsume={removeOne} onEdit={openProduct} />
      : <ProductsPage items={items} onAdd={openNewProduct} onEdit={openProduct} />;

  return (
    <main className="app-shell">
      <div className="app-content">{content}</div>
      <BottomNavigation activePage={activePage} onNavigate={setActivePage} />
      {notice && <div className="toast" role="status">{notice}</div>}
    </main>
  );
}
