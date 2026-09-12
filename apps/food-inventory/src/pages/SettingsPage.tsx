import { Cloud, Copy, RefreshCw, Smartphone, Users } from "lucide-react";
import { useState } from "react";
import type { SyncAccount } from "../services/sync/supabaseSync";

interface SettingsPageProps {
  configured: boolean;
  account: SyncAccount | null;
  onConnectDevice: () => Promise<void>;
  onCreateHousehold: (name: string) => Promise<{ householdId: string; inviteCode: string }>;
  onJoinHousehold: (code: string) => Promise<void>;
  onSync: () => Promise<void>;
}

export function SettingsPage({ configured, account, onConnectDevice, onCreateHousehold, onJoinHousehold, onSync }: SettingsPageProps) {
  const [householdName, setHouseholdName] = useState("Unser Haushalt");
  const [inviteCode, setInviteCode] = useState("");
  const [newInviteCode, setNewInviteCode] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [working, setWorking] = useState(false);

  async function run(action: () => Promise<void>) {
    setWorking(true);
    setMessage(null);
    try { await action(); } catch (error) { setMessage(error instanceof Error ? error.message : "Das hat nicht funktioniert."); }
    finally { setWorking(false); }
  }

  if (!configured) return (
    <section className="page settings-page"><header className="page-header"><p className="eyebrow">Gemeinsamer Vorrat</p><h1>Synchronisierung</h1></header><div className="empty-state"><Cloud aria-hidden="true" size={28} /><p>Die Cloud-Verbindung ist noch nicht eingerichtet.</p></div><p className="muted-status">Die App funktioniert weiterhin vollstaendig lokal.</p></section>
  );

  if (!account) return (
    <section className="page settings-page"><header className="page-header"><p className="eyebrow">Gemeinsamer Vorrat</p><h1>Synchronisierung</h1></header><p className="muted-status">Dieses Geraet erhaelt eine lokale, anonyme Identitaet. Es werden keine E-Mail-Adresse und kein Passwort benoetigt.</p><button className="submit-button" disabled={working} onClick={() => void run(onConnectDevice)} type="button"><Smartphone aria-hidden="true" size={20} />{working ? "Wird verbunden ..." : "Dieses Geraet verbinden"}</button>{message && <p className="sync-message" role="status">{message}</p>}</section>
  );

  if (!account.householdId) return (
    <section className="page settings-page"><header className="page-header"><p className="eyebrow">Dieses Geraet ist verbunden</p><h1>Gemeinsamer Vorrat</h1></header><form className="product-form" onSubmit={(event) => { event.preventDefault(); void run(async () => { const household = await onCreateHousehold(householdName); setNewInviteCode(household.inviteCode); }); }}><label>Name eures Haushalts<input autoComplete="organization" onChange={(event) => setHouseholdName(event.target.value)} required value={householdName} /></label><button className="submit-button" disabled={working || !householdName.trim()} type="submit"><Users aria-hidden="true" size={20} />Haushalt erstellen</button></form><div className="or-divider">oder</div><form className="product-form" onSubmit={(event) => { event.preventDefault(); void run(() => onJoinHousehold(inviteCode)); }}><label>Einladungscode<input autoCapitalize="characters" autoComplete="off" onChange={(event) => setInviteCode(event.target.value)} required value={inviteCode} /></label><button className="manual-action" disabled={working || !inviteCode.trim()} type="submit">Haushalt beitreten</button></form>{newInviteCode && <div className="invite-code"><span>Teile diesen Code einmal mit deiner Frau</span><strong>{newInviteCode}</strong><button aria-label="Einladungscode kopieren" className="icon-button" onClick={() => void navigator.clipboard.writeText(newInviteCode)} type="button"><Copy aria-hidden="true" size={20} /></button></div>}{message && <p className="sync-message" role="status">{message}</p>}</section>
  );

  return (
    <section className="page settings-page"><header className="page-header"><p className="eyebrow">Gemeinsamer Vorrat</p><h1>Synchronisierung</h1></header>{newInviteCode && <div className="invite-code"><span>Einladungscode fuer das zweite Geraet</span><strong>{newInviteCode}</strong><button aria-label="Einladungscode kopieren" className="icon-button" onClick={() => void navigator.clipboard.writeText(newInviteCode)} type="button"><Copy aria-hidden="true" size={20} /></button></div>}<button className="submit-button" disabled={working} onClick={() => void run(onSync)} type="button"><RefreshCw aria-hidden="true" size={20} />Jetzt synchronisieren</button><p className="muted-status">{account.lastSyncedAt ? `Zuletzt synchronisiert: ${new Date(account.lastSyncedAt).toLocaleString("de-DE")}` : "Noch nicht synchronisiert."}</p>{message && <p className="sync-message" role="status">{message}</p>}</section>
  );
}
