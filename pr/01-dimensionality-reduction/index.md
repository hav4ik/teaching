---
title: 1. Снижение размерности данных
layout: chapter
collection: pattern-recognition
order: 1

author:
  name: Chan Kha Vu
  link: https://hav4ik.github.io
date: 24.04.2019
---

[![Open In Colab]({{ site.baseurl }}/assets/badges/colab-badge.svg)](https://colab.research.google.com/drive/1FI2d2_WbCRL9dz03HTkJ85LVvgEmGNL3)
[![Github]({{ site.baseurl }}/assets/badges/github.svg)](#)

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
```

Перед тем как приступить работу с данными, всегда полезно глянуть глазами как оно вообще выглядит. 

{% capture code %}
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
{% endcapture %}
{% capture fig %}
<img src="mnist-samples.png" alt="drawing" width="100%"/>
{% endcapture %}
{% include fig_with_code.html code=code fig=fig id="fig1" %}

Если развернуть изображения в вектора значений пикселей, то возникает натуральный вопрос: какую форму имеет множество таких точек?

> Существует какая-нибудь преобразование или проекция которая наилучшим образом различит наши классы цифр?

Ответом на этот вопрос и служит классы методов PCA (методы главных компонент), LDA (линейные дискриминаторы) и более сложные нелинейные методы, как например T-SNE.

Перед тем как приступать с следующим действиям, неплохо было бы нормализировать наши данные. Ведь они пока лежат в диапазоне `0 .. 255` &mdash; неочень удобный диапазон. Так же, поскольку у нас ограниченное количество вычислительной мощности, будем ограничиваться лишь `6000` семплами датасета.

```python
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
X, Y = images[:6000], Y = labels[:6000]
```

## Метод главных компонент (PCA)

[PCA][pca] (**P**rincipal **C**omponent **A**nalysis) &mdash; один из простейших, но самых распостранённых линейных методов уменьшить [размерности][dimensionality] данных, потеряв при этом наименьшее количество информации.

Пусть $$\boldsymbol{X} = (\boldsymbol{X}_1, \boldsymbol{X}_2, \ldots, \boldsymbol{X}_n)^\intercal$$ &mdash; матрица размером $$n \times d$$ наших изображений, выравненные в вектора.
Посчитаем собственные вектора емпирической матрицы ковариации

$$
\begin{equation} \label{empirical-cov} \tag{1.1}
\mathrm{Cov}[\boldsymbol{X}, \boldsymbol{X}] \approx \frac{1}{n-1} \sum_{i=1}^n {\left(\boldsymbol{X}_i - \bar{\boldsymbol{X}}\right) \left(\boldsymbol{X}_i - \bar{\boldsymbol{X}}\right)^\intercal}
\end{equation}
$$

 и посмотрим на отсортированные собственные значения $$\lambda_i$$:

{% capture code %}
```python
# Calculating Eigenvectors and eigenvalues of Cov matirx
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eigh(cov_mat)

# Sort indices by the descendance of eigenvalues
idx = np.flip(np.argsort(eig_vals))

# Calculation of Explained Variance from the eigenvalues
var_exp = 100 * eig_vals[idx[:100]] / eig_vals.sum() # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
```
{% endcapture %}
{% capture fig %}
{% include_relative eigenvals.html %}
{% endcapture %}
{% include fig_with_code.html code=code fig=fig id="fig2" %}

Можем заметить следующее: собственные значения уменьшаются експоненциально. Это означает, что некоторые направления намного "важнее" других &mdash; если представить множество наших точек (изображений), то такое множество наиболее "растянуто" по этим направлениям.

Давайте посмотрим, как же выглядит собственные вектора $$\boldsymbol{v}_i$$ в порядке уменьшения их собственных значений:

{% capture code %}
```python
n_row, n_col = 4, 8
plt.figure(figsize=(15,8))
for i in range(n_row * n_col):
    plt.subplot(n_row, n_col, i+1)
    plt.imshow(eig_vecs.T[idx][i].reshape(28, 28))
    plt.title('Eigenvector {}'.format(i+1), size=8)
    plt.xticks(()), plt.yticks(())
```
{% endcapture %}
{% capture fig %}
<img src="eigenvecs.png" alt="drawing" width="100%"/>
{% endcapture %}
{% include fig_with_code.html code=code fig=fig id="fig3" %}

Визуально можно увидеть, что вектора с наибольшим собственным значением выглядит как "шаблон" для некоторых цифр, в то время как последние вектора с меньшими собственными значениями визуально не несут никакого семантического значения.

Идея метода главных компонент (PCA) заключается в том, что, посчитав собственные вектора матрицы $$\ref{empirical-cov}$$, спроецировать наши точки $$\boldsymbol{X}$$ на подпространство, порождаемое первыми $$k$$ собственными векторами с наибольшим собственным значением:

$$
\begin{equation}\label{projection} \tag{1.2}
\boldsymbol{Y} = \boldsymbol{X} \times \boldsymbol{W}\,,
\end{equation}
$$

где $$\boldsymbol{W} = (\boldsymbol{v}_{s(1)}, \boldsymbol{v}_{s(2)}, \ldots, \boldsymbol{v}_{s(k)})$$ &mdash; матрица $$k \times d$$ векторов с наибольшими собственными значениями ($$s(i)$$ &mdash; сортировка индексов).

### Визуализация

{% capture code %}
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_std)
X_nd = pca.transform(X_std)
```
{% endcapture %}
{% capture fig %}
{% include_relative pca.html %}
{% endcapture %}
{% include fig_with_code.html code=code fig=fig id="fig7" %}



### Кластеризация методом k-средних

Метод [$$k$$-средних][kmeans] &mdash; один из наиболее популярных методов клаастеризации, зачастую используют вместе с методом главных компонент для выявления возможных кластеров. Данный метод разделяет наше множество точек на $$k$$ *кластеров*, таким образом, чтоб минимизировать дисперцию внутри этих кластеров:

$$
\begin{equation} \label{kmeans-var} \tag{1.3}
\underset{\boldsymbol{S}}{\mathrm{arg\,min}}
\sum_{i=1}^k {
	\sum_{\boldsymbol{x} \in S_i} {| S_i | \mathrm{Var} S_i}}\,,
\end{equation}
$$

где $$S_i$$ &mdash; полученные кластеры, $$i = 1, 2, \ldots, k\,$$. Это эквивалентно минимизации суммарного квадратичного отклонения точек кластеров от центров этих кластеров: 

$$
\begin{equation} \label{kmeans-rms} \tag{1.4}
\underset{\boldsymbol{S}}{\mathrm{arg\,min}}
\sum_{i=1}^k {
	\sum_{\boldsymbol{x} \in S_i} \left\| \boldsymbol{x} - \boldsymbol{\mu}_i \right\|^2
}
\end{equation}
$$

где $$\boldsymbol{\mu}_i$$ обозначает центры масс всех векторов $$\boldsymbol{x}$$ из кластера $$S_i\,$$. Разбиение после применения PCA к нашему набору изображений рукописных символов выглядит следующим образом:

{% capture code %}
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
X_clustered = kmeans.fit_predict(X_std)
```
{% endcapture %}
{% capture fig %}
{% include_relative k-means.html %}
{% endcapture %}
{% include fig_with_code.html code=code fig=fig id="fig4" %}


## Линейный дискриминатор Фишера (LDA)

[LDA][lda] (**L**inear **D**discriminant **A**nalysis) &mdash; один из простейших методов уменьшения размерности данных с учитыванием разметки класса.

### Визуализация

{% capture code %}
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_LDA_2D = lda.fit_transform(X_std, Y)
```
{% endcapture %}
{% capture fig %}
{% include_relative lda.html %}
{% endcapture %}
{% include fig_with_code.html code=code fig=fig id="fig5" %}

## Стохастическое вложение соседей (T-SNE)

### Визуализация

{% capture code %}
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(X_std)
```
{% endcapture %}
{% capture fig %}
{% include_relative t-sne.html %}
{% endcapture %}
{% include fig_with_code.html code=code fig=fig id="fig6" %}

[klyushin]: http://om.univ.kiev.ua/ua/user-15/Pattern
[klyushin-lda]: http://om.univ.kiev.ua/users_upload/15/upload/file/pr_lecture_09.pdf
[pca]: https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%B3%D0%BB%D0%B0%D0%B2%D0%BD%D1%8B%D1%85_%D0%BA%D0%BE%D0%BC%D0%BF%D0%BE%D0%BD%D0%B5%D0%BD%D1%82
[mnist]: http://yann.lecun.com/exdb/mnist/index.html
[dimensionality]: https://ru.wikipedia.org/wiki/%D0%A0%D0%B0%D0%B7%D0%BC%D0%B5%D1%80%D0%BD%D0%BE%D1%81%D1%82%D1%8C_%D0%BF%D1%80%D0%BE%D1%81%D1%82%D1%80%D0%B0%D0%BD%D1%81%D1%82%D0%B2%D0%B0
[kmeans]: https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_k-%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D0%B8%D1%85
[lda]: https://ru.wikipedia.org/wiki/%D0%9B%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D1%8B%D0%B9_%D0%B4%D0%B8%D1%81%D0%BA%D1%80%D0%B8%D0%BC%D0%B8%D0%BD%D0%B0%D0%BD%D1%82%D0%BD%D1%8B%D0%B9_%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7