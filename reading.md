---
layout: reading
title: Japanese reading texts
permalink: /reading/
---

{% assign reading_posts = site.categories.reading %}
<section class="reading-section" aria-labelledby="current-articles-title">
  <div class="reading-section-heading">
    <h2 id="current-articles-title">Current articles</h2>
    <span>Choose a text to read</span>
  </div>
  <div class="reading-list">
{% for post in reading_posts %}
{% unless post.archived %}
<article class="reading-list-item">
  <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
  <p class="reading-meta">{{ post.topic }} / {{ post.level }}{% if post.reading_time_minutes %} / approx. {{ post.reading_time_minutes }} min{% endif %}</p>
  <p>{{ post.summary_en | default: post.summary_de }}</p>
  <p class="reading-grammar">{% for pattern in post.grammar %}<code>{{ pattern }}</code>{% unless forloop.last %} {% endunless %}{% endfor %}</p>
</article>
{% endunless %}
{% endfor %}
</div>
</section>

<section class="reading-archive" aria-labelledby="archive-title">
  <div class="reading-section-heading">
    <h2 id="archive-title">Archive</h2>
    <span>Articles moved here remain available as links</span>
  </div>
  <ul class="archive-list">
{% assign has_archived_articles = false %}
{% for post in reading_posts %}
{% if post.archived %}
{% assign has_archived_articles = true %}
    <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a><span>{{ post.topic }} / {{ post.level }}</span></li>
{% endif %}
{% endfor %}
{% unless has_archived_articles %}
    <li class="archive-empty">No archived articles yet.</li>
{% endunless %}
</ul>
</section>
</div>
