
#!/bin/bash
set -euo pipefail

ITERS=100
REPEATS=5
GRID_SIZES=(256 512 1024 2048)
BLOCK_SIZES=("8 8" "16 16" "32 32")

OUT=results.csv
SUMMARY=summary.csv

echo "version,rows,cols,iters,block_x,block_y,repeat,total_ms,avg_ms_per_gen" > "$OUT"
echo "version,rows,cols,block_x,block_y,mean_ms,min_ms,max_ms" > "$SUMMARY"

run_config() {
    local version="$1"
    local exe="$2"
    local n="$3"
    local bx="$4"
    local by="$5"

    local -a args=(bench "$n" "$n" "$ITERS")
    if [[ "$version" != "cpu" ]]; then
        args+=("$bx" "$by")
    fi

    echo "  $version ${n}x${n} block=${bx}x${by}"

    # Untimed warm-up
    "$exe" "${args[@]}" > /dev/null

    # Five timed repeats; each execution starts from the program's initial grid.
    for rep in $(seq 1 "$REPEATS"); do
        line=$("$exe" "${args[@]}")
        total=$(echo "$line" | grep -oP 'total=\K[0-9.]+' | head -1)
        avg=$(echo "$line" | grep -oP 'avg/gen=\K[0-9.]+' | head -1)

        echo "$version,$n,$n,$ITERS,$bx,$by,$rep,$total,$avg" >> "$OUT"
        echo "    Repeat $rep: total=${total} ms, avg/gen=${avg} ms"
    done

    # Calculate mean and min–max range from the five total times.
    awk -F, -v ver="$version" -v n="$n" -v bx="$bx" -v by="$by" '
        $1==ver && $2==n && $5==bx && $6==by {
            sum += $8
            if (count == 0 || $8 < min) min = $8
            if (count == 0 || $8 > max) max = $8
            count++
        }
        END {
            if (count > 0)
                printf "%s,%s,%s,%s,%s,%.3f,%.3f,%.3f\n",
                    ver,n,n,bx,by,sum/count,min,max
        }
    ' "$OUT" >> "$SUMMARY"
}

for N in "${GRID_SIZES[@]}"; do
    echo "=== Grid ${N}x${N} ==="

    run_config cpu ./cpu/gol_cpu "$N" "" ""

    for bs in "${BLOCK_SIZES[@]}"; do
        read -r bx by <<< "$bs"
        run_config cuda_naive ./cuda/gol_naive "$N" "$bx" "$by"
        run_config cuda_shared ./cuda/gol_shared "$N" "$bx" "$by"
    done
done

echo ""
echo "Benchmark complete."
echo "Individual trials: $OUT"
echo "Mean and range:    $SUMMARY"

