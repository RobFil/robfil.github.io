import { Box, House, PackageSearch, Settings } from "lucide-react";

export type PageId = "home" | "inventory" | "products" | "settings";

const items = [
  { id: "home", label: "Start", icon: House },
  { id: "inventory", label: "Bestand", icon: Box },
  { id: "products", label: "Produkte", icon: PackageSearch },
  { id: "settings", label: "Gemeinsam", icon: Settings },
] as const;

interface BottomNavigationProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
}

export function BottomNavigation({ activePage, onNavigate }: BottomNavigationProps) {
  return (
    <nav aria-label="Hauptnavigation" className="bottom-navigation">
      {items.map((item) => {
        const Icon = item.icon;
        const active = item.id === activePage;
        return (
          <button
            aria-current={active ? "page" : undefined}
            className={active ? "navigation-item active" : "navigation-item"}
            key={item.id}
            onClick={() => onNavigate(item.id)}
            type="button"
          >
            <Icon aria-hidden="true" size={22} strokeWidth={active ? 2.5 : 2} />
            <span>{item.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
