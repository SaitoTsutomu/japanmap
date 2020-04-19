`japanmap` is a package for Japanese map.
::

   import matplotlib.pyplot as plt
   from japanmap import picture, get_data, pref_map
   pct = picture({'北海道': 'blue'})  # numpy.ndarray
   # pct = picture({1: 'blue'})  # same to above
   plt.imshow(pct)  # show graphics
   plt.savefig('map.png')  # save to PNG file
   svg = pref_map(range(1,48), qpqo=get_data())  # IPython.display.SVG
   print(svg.data)  # SVG source

Requirements
------------
* Python 3, Numpy

Features
--------
* nothing

Setup
-----
::

   $ pip install japanmap

History
-------
0.0.1 (2016-6-7)
~~~~~~~~~~~~~~~~~~
* first release
