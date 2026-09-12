export type StorageLocation = "fridge" | "freezer" | "pantry" | "other";
export type ProductSource = "manual" | "open_food_facts";
export type InventoryEventType =
  | "purchase"
  | "consume"
  | "correction"
  | "manual_add"
  | "manual_remove";

export interface Product {
  id: string;
  barcode: string | null;
  name: string;
  genericIngredient: string | null;
  brand: string | null;
  amount: number | null;
  unit: "g" | "kg" | "ml" | "l" | "piece" | "package" | null;
  storageLocation: StorageLocation;
  source: ProductSource;
  createdAt: string;
  updatedAt: string;
}

export interface InventoryEvent {
  id: string;
  productId: string;
  eventType: InventoryEventType;
  quantityChange: number;
  timestamp: string;
  deviceId: string;
  synced: boolean;
}

export interface AppSettings {
  key: "app";
  deviceId: string;
  defaultStorageLocation: StorageLocation;
  vibrationEnabled: boolean;
  soundEnabled: boolean;
}
