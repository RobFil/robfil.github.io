import { Cloud, Copy, LogOut, RefreshCw, Users } from "lucide-react";
import { type FormEvent, useState } from "react";
import type { SyncAccount } from "../services/sync/supabaseSync";

interface SettingsPageProps {
  configured: boolean;
  account: SyncAccount | null;
  onSendLink: (email: string) => Promise<void>;
  onCreateHousehold: (name: string) => Promise<{ householdId: string; inviteCode: string }>;
  onJoinHousehold: (code: string) => Promise<void>;
  onSync: () => Promise<void>;
  onSignOut: () => Promise<void>;
}

export function SettingsPage({ configured, account, onSendLink, onCreateHousehold, onJoinHousehold, onSync, onSignOut }: SettingsPageProps) {
  const [email, setEmail] = useState("");
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

  function submitLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void run(async () => { await onSendLink(email); setMessage("Pruefe dein E-Mail-Postfach und oeffne den Anmeldelink auf diesem Geraet."); });
  }

  if (!configured) return (
    <section className="page settings-page"><header className="page-header"><p className="eyebrow">Gemeinsamer Vorrat</p><h1>Synchronisierung</h1></header><div className="empty-state"><Cloud aria-hidden="true" size={28} /><p>Die Cloud-Verbindung ist noch nicht eingerichtet.</p></div><p className="muted-status">Die App funktioniert weiterhin vollstaendig lokal.</p></section>
  );

  if (!account) return (
    <section className="page settings-page"><header className="page-header"><p className="eyebrow">Gemeinsamer Vorrat</p><h1>Synchronisierung</h1></header><form className="product-form" onSubmit={submitLogin}><label>E-Mail-Adresse<input autoComplete="email" inputMode="email" onChange={(event) => setEmail(event.target.value)} required type="email" value={email} /></label><button className="submit-button" disabled={working} type="submit"><Cloud aria-hidden="true" size={20} />Anmeldelink senden</button></form>{message && <p className="sync-message" role="status">{message}</p>}</section>
  );

  if (!account.householdId) return (
    <section className="page settings-page"><header className="page-header"><p className="eyebrow">Angemeldet als {account.email}</p><h1>Gemeinsamer Vorrat</h1></header><form className="product-form" onSubmit={(event) => { event.preventDefault(); void run(async () => { const household = await onCreateHousehold(householdName); setNewInviteCode(household.inviteCode); }); }}><label>Name eures Haushalts<input autoComplete="organization" onChange={(event) => setHouseholdName(event.target.value)} required value={householdName} /></label><button className="submit-button" disabled={working || !householdName.trim()} type="submit"><Users aria-hidden="true" size={20} />Haushalt erstellen</button></form><div className="or-divider">oder</div><form className="product-form" onSubmit={(event) => { event.preventDefault(); void run(() => onJoinHousehold(inviteCode)); }}><label>Einladungscode<input autoCapitalize="characters" autoComplete="off" onChange={(event) => setInviteCode(event.target.value)} required value={inviteCode} /></label><button className="manual-action" disabled={working || !inviteCode.trim()} type="submit">Haushalt beitreten</button></form>{newInviteCode && <div className="invite-code"><span>Teile diesen Code einmal mit deiner Frau</span><strong>{newInviteCode}</strong><button aria-label="Einladungscode kopieren" className="icon-button" onClick={() => void navigator.clipboard.writeText(newInviteCode)} type="button"><Copy aria-hidden="true" size={20} /></button></div>}{message && <p className="sync-message" role="status">{message}</p>}</section>
  );

  return (
    <section className="page settings-page"><header className="page-header"><p className="eyebrow">Gemeinsamer Vorrat</p><h1>Synchronisierung</h1></header><p className="account-email">{account.email}</p>{newInviteCode && <div className="invite-code"><span>Einladungscode fuer das zweite Geraet</span><strong>{newInviteCode}</strong><button aria-label="Einladungscode kopieren" className="icon-button" onClick={() => void navigator.clipboard.writeText(newInviteCode)} type="button"><Copy aria-hidden="true" size={20} /></button></div>}<button className="submit-button" disabled={working} onClick={() => void run(onSync)} type="button"><RefreshCw aria-hidden="true" size={20} />Jetzt synchronisieren</button><p className="muted-status">{account.lastSyncedAt ? `Zuletzt synchronisiert: ${new Date(account.lastSyncedAt).toLocaleString("de-DE")}` : "Noch nicht synchronisiert."}</p><button className="text-action" disabled={working} onClick={() => void run(onSignOut)} type="button"><LogOut aria-hidden="true" size={18} />Abmelden</button>{message && <p className="sync-message" role="status">{message}</p>}</section>
  );
}
