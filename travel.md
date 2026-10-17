---
layout: page
title: Reisen
permalink: /travel/
---

<p>Eine persönliche Übersicht besuchter Orte in Japan.</p>

<section class="travel-map-callout" aria-label="Karte besuchter Orte">
  <p><strong>Japan Field Log</strong><br>Die Karte öffnet sich als eigenständige, interaktive Ansicht.</p>
  <a class="button-link" href="{{ '/japan_trips.html' | relative_url }}">Karte öffnen <span aria-hidden="true">→</span></a>
</section>

{% assign travel_posts = site.categories.travel %}
{% if travel_posts.size > 0 %}
## Beiträge

<ul>
{% for post in travel_posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> — {{ post.date | date: "%Y-%m-%d" }}</li>
{% endfor %}
</ul>
{% endif %}
