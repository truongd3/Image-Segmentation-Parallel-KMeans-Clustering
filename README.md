# 🖼️ Optimize Image Segmentation with Parallel KMeans Clustering

- Developed **4 versions** of hybrid MPI and CUDA-based KMeans clustering to accelerate image segmentation on **100 pet images**
- Converted each image into up to **2,073,600 RGB pixels** using OpenCV and distributed across **2 GPUs** in C++, parallelizing distance calculations and centroid updates

### Members

- Minh Le
- Truong Dang
- [Dr. Andrew Sohn](https://web.njit.edu/~sohna)

## 💭 Inspiration

### Why Image Segmentation?

Help isolate subjects (like pets in our test images) by _removing background clutter and unnecessary features_. This process also helps improve tasks such as pet **recognition** (object detection), **classification**, or **focused editing**.

### Why KMeans?

A simple data-driven algorithm that works well with multi-dimensional features (3.g. RGB is in 3-dimension).

### Why Parallelize?

As real-world images (Full HD, 4K, Digital Camera) contain millions of pixels, we implement KMeans in plain C++, MPI, and CUDA to compare their performance and learn HPC patterns.

This project is the intersection of computer vision, ML, and HPC, providing an understanding of scalable segmentation.

## 🛠️ How To Run

### 1. Create a build directory and run CMake

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### 2. Run the Executable

**Executables** are located in the `build/bin` folder:

- **Serial version:** `./bin/img_seg_serial`  
- **MPI version:** `./bin/img_seg_mpi`  
- **CUDA version:** `./bin/img_seg_gpu`  
- **MPI + CUDA version:** `./bin/img_seg_mpi_cuda`  

> ℹ️ Navigate to the `build` folder to run the executables.

**Serial & CUDA versions:**

```bash
<executable> <input-path> <num-clusters> <output-path>
```

**MPI & MPI + CUDA versions:**

```bash
mpirun -np <numprocs> <executable> <input-path> <num-clusters> <output-path>
```

### 3. Commands Used to Test

#### Serial

`./bin/img_seg_serial ../db/input/dog48.jpg 5 ../db/output/output_dog48_serial5.jpg`

`./bin/img_seg_serial ../db/input/cat2.jpg 5 ../db/output/output_cat2_serial5.jpg`

`./bin/img_seg_serial ../db/input/dog35.jpg 7 ../db/output/output_dog35_serial7.jpg`

#### MPI

`mpirun -np 4 ./bin/img_seg_mpi ../db/input/dog48.jpg 5 ../db/output/output_dog48_mpi5.jpg`

`mpirun -np 4 ./bin/img_seg_mpi ../db/input/cat2.jpg 5 ../db/output/output_cat2_mpi5.jpg`

`mpirun -np 4 ./bin/img_seg_mpi ../db/input/dog35.jpg 7 ../db/output/output_dog35_mpi7.jpg`

#### CUDA

`./bin/img_seg_gpu ../db/input/dog48.jpg 5 ../db/output/output_dog48_cuda5.jpg`

`./bin/img_seg_gpu ../db/input/cat2.jpg 5 ../db/output/output_cat2_cuda5.jpg`

`./bin/img_seg_gpu ../db/input/dog35.jpg 7 ../db/output/output_dog35_cuda7.jpg`

#### MPI + CUDA

`mpirun -np 4 ./bin/img_seg_mpi_cuda ../db/input/dog48.jpg 5 ../db/output/output_dog48_hybrid5.jpg`

`mpirun -np 4 ./bin/img_seg_mpi_cuda ../db/input/cat2.jpg 5 ../db/output/output_cat2_hybrid5.jpg`

`mpirun -np 4 ./bin/img_seg_mpi_cuda ../db/input/dog35.jpg 7 ../db/output/output_dog35_hybrid7.jpg`

## 💻 Technologies

- C++
- MPI
- CUDA
- OpenCV
- GPUs