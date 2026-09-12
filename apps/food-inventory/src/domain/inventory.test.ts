import { describe, expect, it } from "vitest";
import { getInventoryQuantity } from "./inventory";
import type { InventoryEvent } from "./types";

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

describe("getInventoryQuantity", () => {
  it("sums append-only inventory events", () => {
    expect(
      getInventoryQuantity([
        event(2),
        event(-1),
      ]),
    ).toBe(1);
  });
});
