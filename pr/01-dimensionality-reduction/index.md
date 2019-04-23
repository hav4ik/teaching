---
layout: chapter
collection: pattern-recognition
title: 1. Снижение размерности данных
---

[![Open In Colab](/assets/badges/colab-badge.svg)](https://colab.research.google.com/drive/1FI2d2_WbCRL9dz03HTkJ85LVvgEmGNL3)
[![Open In Colab](/assets/badges/github.svg)](#)

Данный материал является дополнением к [лекциям проф. Клюшин Д. А.][klyushin] по распознаванию образов, а именно &mdash; к лекции о [линейном дискриминаторе Фишера][klyushin-lda]. В данном разделе, мы глубже рассмотрим данный алгоритм на примере проблемы снижения размерности данных базы [MNIST][mnist] вручную написанных цифр, а так же другие методы, а именно:

- Метод главных компонент (PCA) вместе с методом $$k$$-средних
- Линейный дискриминатор Фишера (LDA)
- Стохастическое вложение соседей с $$t$$-распределением (T-SNE)



## База рукописных цифр MNIST

[MNIST][mnist] (**M**odified **N**ational **I**nstitute of **S**tandards and **T**echnology) &mdash; объёмная база данных образцов рукописного написания цифр (изображения размером `(28, 28)`. Является одним из стандатрных датасетов как для тестирования новых алгоритмов, так и для демонстрации методов компьютерного зрения.

### Загрузка

Из-за своей популярности, почти все современные библиотеки для машинного обучения позволяют автоматически загрузить эту базу. Сделаем это с помощью `sklearn.datasets`, после чего сделаем нормализацию данных:

```python
from sklearn.datasets import fetch_openml
images, labels = fetch_openml('mnist_784', version=1, return_X_y=True)

from sklearn.preprocessing import StandardScaler
X, Y = images[:6000], labels[:6000]
X_std = StandardScaler().fit_transform(X)
```

Перед тем как приступить работу с данными, всегда полезно глянуть глазами как оно вообще выглядит. 

<hr class="zero-everything">
<button class="btn btn-primary btn-sm " type="button" data-toggle="collapse" data-target="#collapseMNISTvis" aria-expanded="false" aria-controls="collapseMNISTvis">
	<i class="fas fa-angle-down"></i> показать исходники
</button>
<div class="collapse markdown-1" id="collapseMNISTvis">

```python
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(15,6))
for digit in range(10):
    cols = []
    for col in range(5):
        digit_indices = label_indices[digit][col*5:(col+1)*5]
        cols.append(np.concatenate(images[digit_indices].reshape(5, 28, 28)))
        vis = np.concatenate(cols, axis=1)
    plt.subplot(2, 5, digit + 1), plt.title('Рукописные '+str(digit))
    plt.xticks([]), plt.yticks([])
    plt.imshow(vis, 'gray')
```

</div>

<img src="mnist-samples.png" alt="drawing" width="100%"/>
<hr>

### Собственные значения и вектора


<img src="eigenvecs.png" alt="drawing" width="100%"/>


## Метод главных компонент (PCA)


### Визуализация

<hr class="zero-everything">
<button class="btn btn-primary btn-sm " type="button" data-toggle="collapse" data-target="#collapsePCAvis" aria-expanded="false" aria-controls="collapsePCAvis">
	<i class="fas fa-angle-down"></i> показать исходники
</button>
<div class="collapse markdown-1" id="collapsePCAvis">

```python
pca = PCA(n_components=2)
pca.fit(X_std)
X_nd = pca.transform(X_std)
```

</div>

{% include_relative pca.html %}
<hr>



### Кластеризация методом $$k$$-средних

{% include_relative k-means.html %}

## Линейный дискриминатор Фишера (LDA)

{% include_relative lda.html %}

## Стохастическое вложение соседей (T-SNE)

{% include_relative t-sne.html %}


[klyushin]: http://om.univ.kiev.ua/ua/user-15/Pattern
[klyushin-lda]: http://om.univ.kiev.ua/users_upload/15/upload/file/pr_lecture_09.pdf
[mnist]: http://yann.lecun.com/exdb/mnist/index.html