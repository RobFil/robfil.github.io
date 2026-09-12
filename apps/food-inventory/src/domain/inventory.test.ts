import { describe, expect, it } from "vitest";
import { getInventoryItems, getInventoryQuantity } from "./inventory";
import type { InventoryEvent, Product } from "./types";

function event(quantityChange: number): InventoryEvent {
  return {
    id: `event-${quantityChange}`,
    productId: "product-1",
    eventType: quantityChange >= 0 ? "purchase" : "consume",
    quantityChange,
    timestamp: "2026-09-12T10:00:00.000Z",
    deviceId: "test-device",
    synced: false,
  };
}

function product(id: string, name: string): Product {
  return {
    id,
    barcode: null,
    name,
    genericIngredient: null,
    source: "manual",
    createdAt: "2026-09-12T10:00:00.000Z",
    updatedAt: "2026-09-12T10:00:00.000Z",
  };
}

describe("getInventoryQuantity", () => {
  it("sums append-only inventory events", () => {
    expect(
      getInventoryQuantity([
        event(2),
        event(-1),
      ]),
    ).toBe(1);
  });

  it("derives quantities for every known product", () => {
    const milk = product("milk", "Milch");
    const rice = product("rice", "Reis");

    const items = getInventoryItems([rice, milk], [
      { ...event(2), productId: "milk" },
      { ...event(-1), productId: "milk" },
    ]);

    expect(items).toEqual([
      { product: milk, quantity: 1 },
      { product: rice, quantity: 0 },
    ]);
  });
});
