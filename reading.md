---
layout: page
title: Lesetexte
permalink: /reading/
---

{% assign reading_posts = site.categories.reading %}
{% for post in reading_posts %}
<article>
  <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
  <p>{{ post.topic }} · {{ post.level }}</p>
  <p>{{ post.summary_de }}</p>
  <p>{% for pattern in post.grammar %}<code>{{ pattern }}</code>{% unless forloop.last %} {% endunless %}{% endfor %}</p>
</article>
{% endfor %}
