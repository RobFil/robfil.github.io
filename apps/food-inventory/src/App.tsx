import { useCallback, useEffect, useMemo, useState } from "react";
import { BottomNavigation, type PageId } from "./components/BottomNavigation";
import type { InventoryItem } from "./domain/inventory";
import type { Product } from "./domain/types";
import { HomePage } from "./pages/HomePage";
import { InventoryPage } from "./pages/InventoryPage";
import { ProductFormPage } from "./pages/ProductFormPage";
import { ProductsPage } from "./pages/ProductsPage";
import { ScannerPage, type ScanMode } from "./pages/ScannerPage";
import { SettingsPage } from "./pages/SettingsPage";
import { consumeProduct, createProduct, getAllInventory, getProductByBarcode, purchaseProduct, updateProduct } from "./services/inventoryService";
import { createHousehold, getSyncAccount, isSupabaseConfigured, joinHousehold, onAuthStateChange, sendSignInLink, signOut, syncNow, watchHouseholdChanges, type SyncAccount } from "./services/sync/supabaseSync";

export default function App() {
  const [activePage, setActivePage] = useState<PageId>("home");
  const [items, setItems] = useState<InventoryItem[]>([]);
  const [formProduct, setFormProduct] = useState<Product | null>(null);
  const [formOpen, setFormOpen] = useState(false);
  const [formBarcode, setFormBarcode] = useState<string | undefined>();
  const [scannerMode, setScannerMode] = useState<ScanMode | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [syncAccount, setSyncAccount] = useState<SyncAccount | null>(null);

  const refresh = useCallback(async () => setItems(await getAllInventory()), []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const refreshSyncAccount = useCallback(async () => setSyncAccount(await getSyncAccount()), []);

  const syncInBackground = useCallback(async () => {
    try {
      await syncNow();
      await refreshSyncAccount();
      await refresh();
    } catch {
      // Sync is opportunistic. The local event stays queued for the next connection.
    }
  }, [refresh, refreshSyncAccount]);

  useEffect(() => {
    void refreshSyncAccount();
    const unsubscribe = onAuthStateChange(() => {
      void refreshSyncAccount();
      void syncInBackground();
    });
    window.addEventListener("online", syncInBackground);
    return () => {
      unsubscribe?.();
      window.removeEventListener("online", syncInBackground);
    };
  }, [refreshSyncAccount, syncInBackground]);

  useEffect(() => {
    if (!syncAccount?.householdId) return;
    let unsubscribe: (() => void) | undefined;
    void syncInBackground();
    void watchHouseholdChanges(syncInBackground).then((stop) => { unsubscribe = stop; });
    return () => unsubscribe?.();
  }, [syncAccount?.householdId, syncInBackground]);

  const stockedItemCount = useMemo(() => items.filter((item) => item.quantity > 0).length, [items]);

  function showNotice(message: string) {
    setNotice(message);
    window.setTimeout(() => setNotice(null), 2400);
  }

  function openNewProduct(barcode?: string) {
    setFormProduct(null);
    setFormBarcode(barcode);
    setFormOpen(true);
  }

  function openProduct(productId: string) {
    const item = items.find((entry) => entry.product.id === productId);
    if (!item) return;
    setFormProduct(item.product);
    setFormBarcode(undefined);
    setFormOpen(true);
  }

  async function addOne(productId: string) {
    await purchaseProduct(productId);
    await refresh();
    void syncInBackground();
    showNotice("Zum Bestand hinzugefuegt.");
  }

  async function removeOne(productId: string) {
    if (!(await consumeProduct(productId))) {
      showNotice("Dieser Artikel ist nicht mehr im Bestand.");
      return;
    }
    await refresh();
    void syncInBackground();
    showNotice("Als verbraucht markiert.");
  }

  async function saveProduct(
    input: { name: string; genericIngredient: string | null; barcode?: string },
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
    void syncInBackground();
    setFormOpen(false);
    setFormBarcode(undefined);
  }

  async function handleBarcode(barcode: string) {
    const mode = scannerMode;
    if (!mode) return;
    const product = await getProductByBarcode(barcode);
    setScannerMode(null);

    if (!product) {
      if (mode === "purchase") {
        openNewProduct(barcode);
        showNotice("Neuer Barcode. Bitte Produkt einmal benennen.");
      } else {
        showNotice("Dieser Barcode ist noch nicht bekannt.");
      }
      return;
    }

    if (mode === "purchase") await addOne(product.id);
    else await removeOne(product.id);
  }

  if (scannerMode) {
    return (
      <main className="app-shell">
        <div className="app-content"><ScannerPage mode={scannerMode} onCancel={() => setScannerMode(null)} onDetected={handleBarcode} /></div>
        {notice && <div className="toast" role="status">{notice}</div>}
      </main>
    );
  }

  if (formOpen) {
    return (
      <main className="app-shell">
        <div className="app-content"><ProductFormPage barcode={formBarcode} onCancel={() => setFormOpen(false)} onSave={saveProduct} product={formProduct ?? undefined} /></div>
        {notice && <div className="toast" role="status">{notice}</div>}
      </main>
    );
  }

  const content = activePage === "home"
    ? <HomePage itemCount={stockedItemCount} onConsume={() => setScannerMode("consume")} onManualAdd={openNewProduct} onPurchase={() => setScannerMode("purchase")} />
    : activePage === "inventory"
      ? <InventoryPage items={items} onAdd={addOne} onConsume={removeOne} onEdit={openProduct} />
      : activePage === "products"
        ? <ProductsPage items={items} onAdd={openNewProduct} onEdit={openProduct} />
        : <SettingsPage
            account={syncAccount}
            configured={isSupabaseConfigured()}
            onCreateHousehold={async (name) => { const household = await createHousehold(name); await refreshSyncAccount(); void syncInBackground(); return household; }}
            onJoinHousehold={async (code) => { await joinHousehold(code); await refreshSyncAccount(); await syncInBackground(); }}
            onSendLink={sendSignInLink}
            onSignOut={async () => { await signOut(); await refreshSyncAccount(); }}
            onSync={async () => { await syncNow(); await refreshSyncAccount(); await refresh(); }}
          />;

  return (
    <main className="app-shell">
      <div className="app-content">{content}</div>
      <BottomNavigation activePage={activePage} onNavigate={setActivePage} />
      {notice && <div className="toast" role="status">{notice}</div>}
    </main>
  );
}
