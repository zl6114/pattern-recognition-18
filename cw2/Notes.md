# Note

## Mahalanobis Distance Learning for Person Re-Identification

### Metric Learning

* Run PCA to reduce dimensionality and noise removal
* During training, learn a Mahalanobis metric __M__

* Distance between two samples x_i and x_j is evaluated using equation:

    ![](2018-12-01-15-19-01.png)

### Classification

* In re-identification, we want to recognize a certain person across different non-overlapping camera views.
* Probe Image: Person image selected in one view
* Gallery Image: We want to detect the person in the probe image in the selection of images in the gallery.
* This is achieved by calculating distances between the probe image and all gallery images using a learned metric, and returning those gallery images with the smallest distances as potential matches.