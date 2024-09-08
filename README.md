`japanmap` is a package for Japanese map.

## Usage

```python
import matplotlib.pyplot as plt
from japanmap import picture, get_data, pref_map

pct = picture({'北海道': 'blue'})  # numpy.ndarray
# pct = picture({1: 'blue'})  # same to above
plt.imshow(pct);  # show graphics
# plt.savefig('map.png')  # save to PNG file
```

![](https://raw.githubusercontent.com/SaitoTsutomu/japanmap/master/images/picture.png)

```python
svg = pref_map(range(1,48), qpqo=get_data(), width=4)
# `svg.data` is SVG source
svg
```

![](https://raw.githubusercontent.com/SaitoTsutomu/japanmap/master/images/pref_map.svg)

## Requirements

* Python 3, Pillow, Numpy, Open-CV

## Setup

```sh
$ pip install japanmap
```

## History

* 0.0.1 (2016-6-7): first release
