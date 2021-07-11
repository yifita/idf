input_dir="$1"

run_dirs=(${input_dir}"/Eval_source/File" ${input_dir}"/Eval_target/File")
colors=("ff7aabfa" "fff5cdc9")
prefixes=("source" "target")
for i in 0 1
do
run_dir=${run_dirs[i]}
echo $run_dir
color=${colors[i]}
echo $color
prefix=${prefixes[i]}

readarray -d '' detailplys < <(find $run_dir -name "mesh_mesh_HighRes_[0-9]*.ply" -print0)
readarray -d '' baseplys < <(find $run_dir -name "mesh_*_base*.ply" -print0)


detailply=${detailplys[${#detailplys[@]} - 1]}
baseply=${baseplys[${#baseplys[@]} - 1]}

echo $detailply
echo $baseply

~/.local/bin/Thea/RenderShape -c $color -v 0,0,-1,0,0,3 \
-a 4 $detailply "${run_dir}/${prefix}_detail.jpg" 400 410

~/.local/bin/Thea/RenderShape -c $color -v 0.7,0.1,-1,-0.8,0,1.6 \
-a 4 $detailply "${run_dir}/${prefix}_detail_1.jpg" 400 410

~/.local/bin/Thea/RenderShape -c $color -v 0,0,-1,0,0,3 \
-a 4 $baseply "${run_dir}/${prefix}_base.jpg" 400 410

~/.local/bin/Thea/RenderShape -c $color -v 0.7,0.1,-1,-0.8,0,1.6 \
-a 4 $baseply "${run_dir}/${prefix}_base_1.jpg" 400 410

done

# shorts
# ~/.local/bin/Thea/RenderShape -c $color -v 0.5,0,1,-1.0,0,-2.0 \
# -a 4 $detailply "${run_dir}/${prefix}_detail.jpg" 400 410

# ~/.local/bin/Thea/RenderShape -c $color -v -0.5,0,-1,1.0,0,2.0 \
# -a 4 $detailply "${run_dir}/${prefix}_detail_1.jpg" 400 410

# ~/.local/bin/Thea/RenderShape -c $color -v 0.5,0,1,-1.0,0,-2.0 \
# -a 4 $baseply "${run_dir}/${prefix}_base.jpg" 400 410

# ~/.local/bin/Thea/RenderShape -c $color -v -0.5,0,-1,1.0,0,2.0 \
# -a 4 $baseply "${run_dir}/${prefix}_base_1.jpg" 400 410