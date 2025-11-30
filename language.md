---
layout: page
title: 日本語
permalink: /language/
---

## Beiträge zur Rubrik „Sprache“

{% assign lang_posts = site.categories.language %}
<ul>
{% for post in lang_posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    — {{ post.date | date: "%Y-%m-%d" }}
  </li>
{% endfor %}
</ul>
