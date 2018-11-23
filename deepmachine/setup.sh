SOURCE=${BASH_SOURCE[0]}
KDIR=$(dirname $(realpath $SOURCE))"/layers/mesh_renderer/kernels"
cd $KDIR
bazel build rasterize_triangles_kernel