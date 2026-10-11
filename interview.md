---
layout: page
title: 就職面接
permalink: /interview/
---

日本語での就職面接に向けた準備用のメモです。職務経験の概要から始め、今後はモーター制御、インバーター制御、モデルベース開発、技術評価などのテーマを個別の中間テキストとして追加します。

## 中間テキスト

{% assign interview_posts = site.categories.interview %}
<ul>
{% for post in interview_posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    — {{ post.date | date: "%Y-%m-%d" }}
    {% if post.summary_de %}<br><small>{{ post.summary_de }}</small>{% endif %}
  </li>
{% endfor %}
</ul>
