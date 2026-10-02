set -e
ITERS=100
GRID_SIZES=(256 512 1024 2048)
BLOCK_SIZES=("8 8" "16 16" "32 32")

OUT=results.csv
echo "version,rows,cols,iters,block_x,block_y,total_ms,avg_ms_per_gen" > "$OUT"

for N in "${GRID_SIZES[@]}"; do
    echo "=== Grid ${N}x${N} ==="

    # CPU (no block size concept, run once per grid size)
    line=$(./cpu/gol_cpu bench "$N" "$N" "$ITERS")
    echo "$line"
    total=$(echo "$line" | grep -oP 'total=\K[0-9.]+')
    avg=$(echo "$line" | grep -oP 'avg/gen=\K[0-9.]+')
    echo "cpu,$N,$N,$ITERS,,,${total},${avg}" >> "$OUT"

    for bs in "${BLOCK_SIZES[@]}"; do
        read -r bx by <<< "$bs"

        line=$(./cuda/gol_naive bench "$N" "$N" "$ITERS" "$bx" "$by")
        echo "$line"
        total=$(echo "$line" | grep -oP 'total=\K[0-9.]+')
        avg=$(echo "$line" | grep -oP 'avg/gen=\K[0-9.]+')
        echo "cuda_naive,$N,$N,$ITERS,$bx,$by,${total},${avg}" >> "$OUT"

        line=$(./cuda/gol_shared bench "$N" "$N" "$ITERS" "$bx" "$by")
        echo "$line"
        total=$(echo "$line" | grep -oP 'total=\K[0-9.]+')
        avg=$(echo "$line" | grep -oP 'avg/gen=\K[0-9.]+')
        echo "cuda_shared,$N,$N,$ITERS,$bx,$by,${total},${avg}" >> "$OUT"
    done
done

echo ""
echo "Done. Results written to $OUT"
