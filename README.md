# NurieTransform
NurieTransform is a Python library for applying filters to images. Filtering is used in a wide range of applications such as segmentation, contour detection, and preprocessing for deep learning.<br>
<br>
The following example shows how NurieTransform can be used for segmenting HeLa cells.<br>

![example](https://github.com/TANEO-bio/NurieTransform/blob/main/example.png)

photo by https://www.biomol.com/resources/biomol-blog/henrietta-lacks-immortal-cells <br>

# A simple example
```python
image = cv2.imread('HeLa.jpg', 0)
filters = [
           nurie.HighpassFilter(),
           nurie.AverageFilter(kernel_size=10),
           nurie.Binalize(threshold=2)
          ]
model = nurie.Sequential(filters=filters)
result = model.fit(image)
```

# List of filters
- [AverageFilter](https://github.com/TANEO-bio/NurieTransform/blob/main/nurie/filters/AverageFilter.py)
- [BilateralFilter](https://github.com/TANEO-bio/NurieTransform/blob/main/nurie/filters/BilateralFilter.py)
- [Binalize](https://github.com/TANEO-bio/NurieTransform/blob/main/nurie/filters/Binalize.py)
- [GaussianBlur](https://github.com/TANEO-bio/NurieTransform/blob/main/nurie/filters/AverageFilter.py)
- [GradientFilter](https://github.com/TANEO-bio/NurieTransform/blob/main/nurie/filters/GradientFilter.py)
- [HighpassFilter](https://github.com/TANEO-bio/NurieTransform/blob/main/nurie/filters/HighpassFilter.py)
- [LaplacianFilter](https://github.com/TANEO-bio/NurieTransform/blob/main/nurie/filters/LaplacianFilter.py)
- [MedianBlur](https://github.com/TANEO-bio/NurieTransform/blob/main/nurie/filters/MedianBlur.py)
- [NNFilter](https://github.com/TANEO-bio/NurieTransform/blob/main/nurie/filters/NNFilter.py)
- [SobelFilter](https://github.com/TANEO-bio/NurieTransform/blob/main/nurie/filters/SobelFilter.py)
