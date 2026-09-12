import { useState } from "react";
import { BottomNavigation, type PageId } from "./components/BottomNavigation";
import { HomePage } from "./pages/HomePage";
import { PlaceholderPage } from "./pages/PlaceholderPage";

const pageContent: Record<Exclude<PageId, "home">, { title: string; description: string }> = {
  inventory: { title: "Bestand", description: "Dein aktueller Vorrat erscheint hier." },
  products: { title: "Produkte", description: "Bekannte Produkte werden hier verwaltet." },
  settings: { title: "Einstellungen", description: "Geraete- und App-Einstellungen werden hier konfiguriert." },
};

export default function App() {
  const [activePage, setActivePage] = useState<PageId>("home");
  const content = activePage === "home" ? <HomePage /> : <PlaceholderPage {...pageContent[activePage]} />;

  return (
    <main className="app-shell">
      <div className="app-content">{content}</div>
      <BottomNavigation activePage={activePage} onNavigate={setActivePage} />
    </main>
  );
}
