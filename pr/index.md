---
title: 0. Введение
layout: chapter
collection: pattern-recognition
order: 0

author:
  name: Chan Kha Vu
  link: https://hav4ik.github.io
date: 24.04.2019
---

## Содержание:

{% assign lectures = site.pages | where: 'collection', 'pattern-recognition' %}
{% assign lectures_sorted = lectures | sort: 'order' %}

<ul>
{% for p in lectures_sorted %}
  <li><a href="{{ site.baseurl }}{{ p.url }}">{{ p.title }}</a></li>
{% endfor %}
</ul>



