# Image Loading Self-check
### ENV1
Alibaba CPFS (120GB/s) cluster, single process, 2048 Laion images
### Test1
1. Raw Loading 
batch size = 1, 0.53s
2. Training-like Loading
batch size = 16, num workers = 8, resize to 256*256, to cpu tensor.
Finally get 1.53s
3. Training High-resolution Images (laion_high_resolution_imgs > 1024)
batch size = 16, num workers = 8, resize to 256*256, to cpu tensor.
6s
batch size = 16, num workers = 1, resize to 256*256, to cpu tensor.
39s
### Conclusion
Using Webdataset, the data loading will not become bottleneck for training if the storage is fast enough. Even for large image, we can achieve ~350 images per sec for a rank.