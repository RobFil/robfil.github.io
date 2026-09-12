import type { InventoryEvent, Product } from "./types";

export interface InventoryItem {
  product: Product;
  quantity: number;
}

export function getInventoryQuantity(events: InventoryEvent[]): number {
  return events.reduce((quantity, event) => quantity + event.quantityChange, 0);
}

export function getInventoryItems(products: Product[], events: InventoryEvent[]): InventoryItem[] {
  const quantities = new Map<string, number>();

  for (const event of events) {
    quantities.set(event.productId, (quantities.get(event.productId) ?? 0) + event.quantityChange);
  }

  return products
    .map((product) => ({ product, quantity: quantities.get(product.id) ?? 0 }))
    .sort((left, right) => left.product.name.localeCompare(right.product.name, "de"));
}
