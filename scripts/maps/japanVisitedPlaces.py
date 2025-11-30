import folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os

# -------------------------------
# Pfad der TXT-Datei relativ zum Skript bestimmen
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "cities.txt")

# -------------------------------
# TXT-Liste einlesen
# -------------------------------
places = []
with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        name = line.strip()
        if name:
            places.append({"name": name})

print(f"{len(places)} Orte geladen aus {filename}")

# -------------------------------
# Geocoding vorbereiten
# -------------------------------
geolocator = Nominatim(user_agent="my-geo-script")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

print("Bestimme Koordinaten...")

for place in places:
    query = f'{place["name"]}, Japan'
    location = geocode(query)

    if location is None:
        print(f'⚠️ Konnte "{place["name"]}" nicht finden')
        place["lat"] = None
        place["lon"] = None
    else:
        place["lat"] = location.latitude
        place["lon"] = location.longitude
        print(f'{place["name"]}: {place["lat"]}, {place["lon"]}')

# -------------------------------
# Karte erstellen
# -------------------------------
m = folium.Map(location=[36.0, 138.0], zoom_start=5)

for p in places:
    if p["lat"] is None:
        continue
    folium.Marker(
        location=[p["lat"], p["lon"]],
        popup=p["name"],
    ).add_to(m)

m.save("japan_trips.html")
print("Fertig: japan_trips.html")
