---
title: 0. Вступ
layout: chapter
collection: pattern-recognition
order: 0
---

{% assign lectures = site.pages | where: 'collection', 'pattern-recognition' %}
{% assign lectures_sorted = lectures | sort: 'order' %}

<ul>
{% for p in lectures_sorted %}
  <li><a href="{{ site.baseurl }}{{ p.url }}">{{ p.title }}</a></li>
{% endfor %}
</ul>
