# 🖼️ Image Segmentation

---

## 🛠️ Building the Project

1. **Make sure OpenCV is installed**  
   Ensure OpenCV is installed on your system via your package manager (`apt`, `brew`, `vcpkg`, etc.), or build it from source.

2. **Create a build directory and run CMake:**

   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

3. **Run the executable**

   - Serial version: `build/seg_cpu`  
   - Usage:

     ```bash
     ./seg_cpu <input-image> <num-clusters> <output-image>
     ```
