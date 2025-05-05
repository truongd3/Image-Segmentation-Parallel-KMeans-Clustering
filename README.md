# 🖼️ Image Segmentation

---

## 🛠️ Building the Project

### 1. Create a build directory and run CMake

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

---

### 2. Run the Executable

Executables are located in the `build/bin` folder:

- **Serial version:** `build/bin/img_seg_serial`  
- **MPI version:** `build/bin/img_seg_mpi`  
- **CUDA version:** `build/bin/img_seg_gpu`  
- **MPI + CUDA version:** `build/bin/img_seg_mpi_cuda`  

> ℹ️ Navigate to the `build/bin` folder to run the executables.

#### ▶️ Usage

**Serial & CUDA versions:**

```bash
./<executable> <input-path> <num-clusters> <output-path>
```

**MPI & MPI + CUDA versions:**

```bash
mpirun -np <numprocs> ./<executable> <input-path> <num-clusters> <output-path>
```

