import type { InventoryEvent, Product } from "../domain/types";

export interface ProductRepository {
  getById(id: string): Promise<Product | undefined>;
  getByBarcode(barcode: string): Promise<Product | undefined>;
  search(query: string): Promise<Product[]>;
  getAll(): Promise<Product[]>;
  save(product: Product): Promise<string>;
  update(product: Product): Promise<string>;
}

export interface InventoryEventRepository {
  add(event: InventoryEvent): Promise<string>;
  getByProduct(productId: string): Promise<InventoryEvent[]>;
  getUnsynced(): Promise<InventoryEvent[]>;
  markSynced(ids: string[]): Promise<number>;
  getAll(): Promise<InventoryEvent[]>;
}
