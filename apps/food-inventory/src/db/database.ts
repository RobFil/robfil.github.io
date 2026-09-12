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
  }
}

export const db = new FoodInventoryDatabase();
