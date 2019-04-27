---
title: "2. SVM: классификация и детекция"
layout: chapter
collection: pattern-recognition
order: 2

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


<div id="hogvis_image"></div>
{% include_relative hog_vis.html %}

Интерактивная демонстрация гистограммы направленных градиентов (HOG) на примере знаменитой в круге специалистов компьютерного зрения фотографии ["Lenna"][lenna].
<hr>

[![Open In Colab]({{ site.baseurl }}/assets/badges/colab-badge.svg)](#)
[![Github]({{ site.baseurl }}/assets/badges/github.svg)](#)

> Данный материал является дополнением к [лекциям проф. Клюшин Д. А.][klyushin] по распознаванию образов, а именно &mdash; к лекции о [методе опорных векторов (SVM)][klyushin-svm].

Один из первых вопросах студента после изучения [метода опорных векторов (SVM)][svm] &mdash; как его применять к разным практическим задачам? К задачам компьютерного зрения или анализа текста? В данном разделе, мы разберём классический пример применения метода SVM для распознавания, а потом и для детекции лица на изображении.

## База размеченных фотографии лиц (LFW)

**LFW** (**L**abeled **F**aces in the **W**ild) &mdash; классическая база данных для распознавания лиц. Содержит `13233` изображении размером `47x62` пикселей.

### Загрузка и обработка

Как и многие стандартные игрушечные датасеты для тестирования и демонстрации методов распознавания образов, данную базу можно загрузить используя `sklearn.datasets`:

```python
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people()
```



## Гистограмма направленных градиентов (HOG)


## Метод опорных векторов (SVM)


## Подавление не-максимумов (NMS pooling)




[lenna]: https://en.wikipedia.org/wiki/Lenna
[klyushin]: http://om.univ.kiev.ua/ua/user-15/Pattern
[klyushin-svm]: http://om.univ.kiev.ua/users_upload/15/upload/file/pr_lecture_07.pdf
[svm]: https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BE%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D1%85_%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BE%D0%B2