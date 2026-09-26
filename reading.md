---
layout: reading
title: Lesetexte
permalink: /reading/
---

<p><a href="{{ '/reading-checker/' | relative_url }}">Programmatischer Reading Checker</a> – Lesetext lokal laden, Satz für Satz sprechen und deterministisch vergleichen.</p>

{% assign reading_posts = site.categories.reading %}
<div class="reading-list">
{% for post in reading_posts %}
<article class="reading-list-item">
  <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
  <p class="reading-meta">{{ post.topic }} / {{ post.level }}{% if post.reading_time_minutes %} / ca. {{ post.reading_time_minutes }} Min.{% endif %}</p>
  <p>{{ post.summary_de }}</p>
  <p class="reading-grammar">{% for pattern in post.grammar %}<code>{{ pattern }}</code>{% unless forloop.last %} {% endunless %}{% endfor %}</p>
</article>
{% endfor %}
</div>
