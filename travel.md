---
layout: page
title: 旅行
permalink: /travel/
---

<p>これまで日本で訪れた場所の個人的な記録です。</p>

<section class="travel-map-callout" aria-label="訪問した場所の地図">
  <p><strong>日本旅行ログ</strong><br>地図は別のインタラクティブなページで開きます。</p>
  <a class="button-link" href="{{ '/japan_trips.html' | relative_url }}">地図を見る <span aria-hidden="true">→</span></a>
</section>

{% assign travel_posts = site.categories.travel %}
{% if travel_posts.size > 0 %}
## 記事

<ul>
{% for post in travel_posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> — {{ post.date | date: "%Y-%m-%d" }}</li>
{% endfor %}
</ul>
{% endif %}
