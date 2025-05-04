# 🖼️ Image Segmentation

---

## 🛠️ Building the Project

1. **Create a build directory and run CMake:**

   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

2. **Run the executable**
   - Executables are located in build/bin folder
   - **Serial version**: `build/bin/img_seg_serial`  
   - **MPI Version**: `build/bin/img_seg_mpi`
   - **CUDA Version**: `build/bin/img_seg_gpu`
   - **MPI + CUDA Version**: `build/bin/img_seg_mpi_cuda`
   - Usage: navigate to build/bin folder to find executables

     ```bash
     ./<executable> <input-path> <num-clusters> <output-path>
     ```
