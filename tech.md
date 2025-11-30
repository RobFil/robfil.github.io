---
layout: page
title: Technik
permalink: /tech/
---

## Beiträge zur Rubrik „Technik“

{% assign tech_posts = site.categories.tech %}
<ul>
{% for post in tech_posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    — {{ post.date | date: "%Y-%m-%d" }}
  </li>
{% endfor %}
</ul>
