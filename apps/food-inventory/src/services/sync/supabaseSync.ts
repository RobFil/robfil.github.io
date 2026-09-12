import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { db } from "../../db/database";
import type { InventoryEvent, Product } from "../../domain/types";
import { getAppSettings, updateAppSettings } from "../inventoryService";

interface RemoteProduct {
  id: string;
  household_id: string;
  barcode: string | null;
  name: string;
  generic_ingredient: string | null;
  source: Product["source"];
  created_at: string;
  updated_at: string;
}

interface RemoteEvent {
  id: string;
  household_id: string;
  product_id: string;
  event_type: InventoryEvent["eventType"];
  quantity_change: number;
  timestamp: string;
  device_id: string;
}

export interface SyncAccount {
  householdId?: string;
  householdName?: string;
  lastSyncedAt?: string;
}

let activeSync: Promise<void> | null = null;
let supabaseClient: SupabaseClient | null | undefined;

function client(): SupabaseClient | null {
  if (supabaseClient !== undefined) return supabaseClient;
  const url = import.meta.env.VITE_SUPABASE_URL;
  const key = import.meta.env.VITE_SUPABASE_ANON_KEY;
  supabaseClient = url && key ? createClient(url, key) : null;
  return supabaseClient;
}

function toRemoteProduct(product: Product, householdId: string): RemoteProduct {
  return {
    id: product.id, household_id: householdId, barcode: product.barcode, name: product.name,
    generic_ingredient: product.genericIngredient, source: product.source,
    created_at: product.createdAt, updated_at: product.updatedAt,
  };
}

function fromRemoteProduct(product: RemoteProduct): Product {
  return {
    id: product.id, barcode: product.barcode, name: product.name,
    genericIngredient: product.generic_ingredient, source: product.source,
    createdAt: product.created_at, updatedAt: product.updated_at,
  };
}

function toRemoteEvent(event: InventoryEvent, householdId: string): RemoteEvent {
  return {
    id: event.id, household_id: householdId, product_id: event.productId,
    event_type: event.eventType, quantity_change: event.quantityChange,
    timestamp: event.timestamp, device_id: event.deviceId,
  };
}

function fromRemoteEvent(event: RemoteEvent): InventoryEvent {
  return {
    id: event.id, productId: event.product_id, eventType: event.event_type,
    quantityChange: event.quantity_change, timestamp: event.timestamp,
    deviceId: event.device_id, synced: true,
  };
}

export function isSupabaseConfigured(): boolean {
  return client() !== null;
}

export async function getSyncAccount(): Promise<SyncAccount | null> {
  const supabase = client();
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  if (!data.session?.user) return null;
  const settings = await getAppSettings();
  return { householdId: settings.householdId, householdName: settings.householdName, lastSyncedAt: settings.lastSyncedAt };
}

export async function connectDevice(): Promise<void> {
  const supabase = client();
  if (!supabase) throw new Error("Supabase ist noch nicht konfiguriert.");
  const { error } = await supabase.auth.signInAnonymously();
  if (error) throw error;
}

export async function createHousehold(name: string): Promise<{ householdId: string; inviteCode: string }> {
  const supabase = client();
  if (!supabase) throw new Error("Supabase ist noch nicht konfiguriert.");
  const { data, error } = await supabase.rpc("create_household", { household_name: name.trim() });
  if (error) throw error;
  const household = (data as Array<{ household_id: string; invite_code: string }>)[0];
  if (!household) throw new Error("Der Haushalt konnte nicht erstellt werden.");
  await updateAppSettings({ householdId: household.household_id, householdName: name.trim() });
  return { householdId: household.household_id, inviteCode: household.invite_code };
}

export async function joinHousehold(inviteCode: string, householdName: string): Promise<void> {
  const supabase = client();
  if (!supabase) throw new Error("Supabase ist noch nicht konfiguriert.");
  const { data, error } = await supabase.rpc("join_household", { household_invite_code: inviteCode.trim().toUpperCase() });
  if (error) throw error;
  const householdId = data as string;
  await updateAppSettings({ householdId, householdName: householdName.trim() || "Gemeinsamer Vorrat" });
}

async function requireReady(): Promise<{ supabase: SupabaseClient; householdId: string }> {
  const supabase = client();
  if (!supabase) throw new Error("Supabase ist noch nicht konfiguriert.");
  const account = await getSyncAccount();
  if (!account) throw new Error("Bitte zuerst anmelden.");
  if (!account.householdId) throw new Error("Bitte zuerst einen Haushalt erstellen oder beitreten.");
  return { supabase, householdId: account.householdId };
}

async function performSync(): Promise<void> {
  const { supabase, householdId } = await requireReady();
  const [{ data: remoteProducts, error: productsError }, { data: remoteEvents, error: eventsError }] = await Promise.all([
    supabase.from("products").select("*").eq("household_id", householdId),
    supabase.from("inventory_events").select("*").eq("household_id", householdId),
  ]);
  if (productsError) throw productsError;
  if (eventsError) throw eventsError;

  const localProducts = await db.products.toArray();
  const remoteProductMap = new Map((remoteProducts as RemoteProduct[]).map((product) => [product.id, product]));
  const productsToPush = localProducts.filter((product) => {
    const remote = remoteProductMap.get(product.id);
    return !remote || product.updatedAt > remote.updated_at;
  });

  await db.transaction("rw", db.products, db.events, async () => {
    for (const remote of remoteProducts as RemoteProduct[]) {
      const local = await db.products.get(remote.id);
      if (!local || remote.updated_at > local.updatedAt) await db.products.put(fromRemoteProduct(remote));
    }
    const remoteEventIds = new Set((remoteEvents as RemoteEvent[]).map((event) => event.id));
    for (const remote of remoteEvents as RemoteEvent[]) {
      if (!(await db.events.get(remote.id))) await db.events.add(fromRemoteEvent(remote));
    }
    const localEvents = await db.events.toArray();
    await db.events.bulkPut(localEvents.map((event) => ({ ...event, synced: remoteEventIds.has(event.id) })));
  });

  if (productsToPush.length) {
    const { error } = await supabase.from("products").upsert(productsToPush.map((product) => toRemoteProduct(product, householdId)));
    if (error) throw error;
  }
  const eventsToPush = await db.events.filter((event) => !event.synced).toArray();
  if (eventsToPush.length) {
    const { error } = await supabase.from("inventory_events").upsert(eventsToPush.map((event) => toRemoteEvent(event, householdId)), { onConflict: "id" });
    if (error) throw error;
    await db.events.bulkPut(eventsToPush.map((event) => ({ ...event, synced: true })));
  }
  await updateAppSettings({ lastSyncedAt: new Date().toISOString() });
}

export function syncNow(): Promise<void> {
  if (activeSync) return activeSync;
  activeSync = performSync().finally(() => { activeSync = null; });
  return activeSync;
}

export async function watchHouseholdChanges(onChange: () => void): Promise<(() => void) | undefined> {
  const { supabase, householdId } = await requireReady();
  const channel = supabase
    .channel(`inventory-${householdId}`)
    .on("postgres_changes", { event: "*", schema: "public", table: "products", filter: `household_id=eq.${householdId}` }, onChange)
    .on("postgres_changes", { event: "*", schema: "public", table: "inventory_events", filter: `household_id=eq.${householdId}` }, onChange)
    .subscribe();
  return () => { void supabase.removeChannel(channel); };
}
