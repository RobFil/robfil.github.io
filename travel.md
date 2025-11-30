---
layout: page
title: Reisen
permalink: /travel/
---

## Beiträge zur Rubrik „Reisen“

{% assign travel_posts = site.categories.travel %}
<ul>
{% for post in travel_posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    — {{ post.date | date: "%Y-%m-%d" }}
  </li>
{% endfor %}
</ul>
