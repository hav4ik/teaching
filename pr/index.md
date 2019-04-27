---
title: Курс распознавание образов
layout: chapter
collection: pattern-recognition
order: 0

author:
  name: Chan Kha Vu
  link: https://hav4ik.github.io
course:
  name: Распознавание образов (ОМ-5)
  link: http://om.univ.kiev.ua/ua/user-15/Pattern
lector:
  name: Клюшин Д. А.
  link: http://om.univ.kiev.ua/ua/user-15
---
> Данная коллекция материалов является дополнением к [лекциям проф. Клюшин Д. А.][klyushin] по распознаванию образов для студентов 1-го курса магистратуры прикладной математики.

## Введение

В рамках [лекции этого курса][klyushin], рассматривались преимущественно классические методы для распознавания образов. В данной серии дополнительного материала, мы более детально рассмотрим использование некоторых из этим методов на практических примерах.

Так же, с целью облегчения осваивания материала, большинство графиков и визуализации в данной серии являются **интерактивными** &mdash; не поленитесь навести на них мышкой.


## Содержание

{% assign lectures = site.pages | where: 'collection', 'pattern-recognition' %}
{% assign lectures_sorted = lectures | sort: 'order' %}

<ul>
{% for p in lectures_sorted %}
  <li><a href="{{ site.baseurl }}{{ p.url }}">{{ p.title }}</a></li>
{% endfor %}
</ul>


## Релевантность

Несмотря на нынешнюю популярность методов обучения глубоких свёрточных нейронных сетей, классические методы распознавания образов всё равно остаются релевантными по нескольким причинам:

- **Когда очень мало данных**, простая регрессия может лучше сработать чем многослойная рекуррентная нейронка.
- Зачастую, классические методы используют для [**автоматической очистки данных**][data-clensing] при автономном сборе большого датасета для обучения глубоких сетей. Знание классических методов в таком случаи может вам сэкономить огромное количество денег на [лейблеров][labeler].


## Обозначения

В дополнительных материалах можете встретить следующие обозначения:

- ![Colab badge]({{ site.baseurl }}/assets/badges/colab-badge.svg) &mdash; есть ссылка на код в формате *Jupyter Notebook* на Google Colab.
- ![Colab badge]({{ site.baseurl }}/assets/badges/github.svg) &mdash; код доступен по ссылке на Github.
- ![GitHub issues](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg) &mdash; ссылка на Github Issues. Если заметили ошибку, пишите туда.
- ![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg) &mdash; тип лицензии, под которым распостраняется код и визуализации.



[klyushin]: http://om.univ.kiev.ua/ua/user-15/Pattern
[data-clensing]: https://en.wikipedia.org/wiki/Data_cleansing
[labeler]: https://towardsdatascience.com/do-you-know-what-does-a-data-labeler-do-98561cb0029