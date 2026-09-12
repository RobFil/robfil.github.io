import type { InventoryEvent } from "./types";

export function getInventoryQuantity(events: InventoryEvent[]): number {
  return events.reduce((quantity, event) => quantity + event.quantityChange, 0);
}
