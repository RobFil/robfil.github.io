---
layout: page
title: Reisen
permalink: /travel/
---

## Beiträge zur Rubrik „Reisen“

{% assign travel_posts = site.posts | where_exp:'post','post.categories contains "travel"' %}
<ul>
{% for post in travel_posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    — {{ post.date | date: "%Y-%m-%d" }}
  </li>
{% endfor %}
</ul>
