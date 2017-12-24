# Image-Stitching

##### Reference https://github.com/kushalvyas/Python-Multiple-Image-Stitching

##### usage: python pano.py [file_list_location] (ex: python pano.py txtlists/files2.txt)


## How to use seam filling

```python
imfile = 'images/tv2.jpg'
im = cv2.imread(imfile)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = utils.im2float(im)
im = imutils.resize(im, width=400) # smaller size for quickly computing

# fill_DIR means fill along DIR direction
# k: the max number of filling iteration, each iteration will add one seam with one pixel width.
# bound: the bound to be sampled (pick appropriate one by youself)
result = sf.fill_right(im, [0, 350], k=40)
```

