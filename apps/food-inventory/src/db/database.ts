import Dexie, { type EntityTable } from "dexie";
import type { AppSettings, InventoryEvent, Product } from "../domain/types";

class FoodInventoryDatabase extends Dexie {
  products!: EntityTable<Product, "id">;
  events!: EntityTable<InventoryEvent, "id">;
  settings!: EntityTable<AppSettings, "key">;

  constructor() {
    super("FoodInventoryDB");
    this.version(1).stores({
      products: "id, &barcode, name, genericIngredient, storageLocation",
      events: "id, productId, timestamp, synced",
      settings: "key",
    });
    this.version(2)
      .stores({
        products: "id, &barcode, name, genericIngredient",
        events: "id, productId, timestamp, synced",
        settings: "key",
      })
      .upgrade(async (transaction) => {
        await transaction.table("products").toCollection().modify((product) => {
          const legacyProduct = product as Record<string, unknown>;
          delete legacyProduct.storageLocation;
          delete legacyProduct.brand;
          delete legacyProduct.amount;
          delete legacyProduct.unit;
        });
        await transaction.table("settings").toCollection().modify((settings) => {
          const legacySettings = settings as Record<string, unknown>;
          delete legacySettings.defaultStorageLocation;
          delete legacySettings.vibrationEnabled;
          delete legacySettings.soundEnabled;
        });
      });
  }
}

export const db = new FoodInventoryDatabase();
