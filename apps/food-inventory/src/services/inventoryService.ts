import { db } from "../db/database";
import { getInventoryItems, getInventoryQuantity, type InventoryItem } from "../domain/inventory";
import type { AppSettings, InventoryEvent, Product } from "../domain/types";

export interface ProductInput {
  name: string;
  genericIngredient: string | null;
  barcode?: string | null;
}

function id(): string {
  return crypto.randomUUID();
}

function now(): string {
  return new Date().toISOString();
}

async function settings(): Promise<AppSettings> {
  const existing = await db.settings.get("app");
  if (existing) return existing;

  const created: AppSettings = {
    key: "app",
    deviceId: id(),
  };
  await db.settings.put(created);
  return created;
}

export async function getAppSettings(): Promise<AppSettings> {
  return settings();
}

export async function updateAppSettings(changes: Partial<Omit<AppSettings, "key" | "deviceId">>): Promise<AppSettings> {
  const current = await settings();
  const updated = { ...current, ...changes };
  await db.settings.put(updated);
  return updated;
}

async function addEvent(productId: string, eventType: InventoryEvent["eventType"], quantityChange: number) {
  const appSettings = await settings();
  await db.events.add({
    id: id(),
    productId,
    eventType,
    quantityChange,
    timestamp: now(),
    deviceId: appSettings.deviceId,
    synced: false,
  });
}

export async function getAllInventory(): Promise<InventoryItem[]> {
  const [products, events] = await Promise.all([db.products.toArray(), db.events.toArray()]);
  return getInventoryItems(products, events);
}

export async function getProductByBarcode(barcode: string): Promise<Product | undefined> {
  return db.products.where("barcode").equals(barcode).first();
}

export async function getStockedItemCount(): Promise<number> {
  const inventory = await getAllInventory();
  return inventory.filter((item) => item.quantity > 0).length;
}

export async function createProduct(input: ProductInput, addToStock: boolean): Promise<Product> {
  const timestamp = now();
  const product: Product = {
    // A barcode is a shared product identity, so both phones create the same ID offline.
    id: input.barcode ? `barcode:${input.barcode}` : id(),
    barcode: input.barcode ?? null,
    name: input.name.trim(),
    genericIngredient: input.genericIngredient?.trim() || null,
    source: "manual",
    createdAt: timestamp,
    updatedAt: timestamp,
  };

  await db.transaction("rw", db.products, db.events, db.settings, async () => {
    await db.products.add(product);
    if (addToStock) await addEvent(product.id, "manual_add", 1);
  });

  return product;
}

export async function updateProduct(product: Product, input: ProductInput): Promise<void> {
  await db.products.put({
    ...product,
    name: input.name.trim(),
    genericIngredient: input.genericIngredient?.trim() || null,
    updatedAt: now(),
  });
}

export async function purchaseProduct(productId: string): Promise<void> {
  await addEvent(productId, "purchase", 1);
}

export async function consumeProduct(productId: string): Promise<boolean> {
  const events = await db.events.where("productId").equals(productId).toArray();
  if (getInventoryQuantity(events) <= 0) return false;
  await addEvent(productId, "consume", -1);
  return true;
}
