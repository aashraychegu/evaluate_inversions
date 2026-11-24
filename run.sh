

run_analysis(){
    # echo "---------------------- 3 ----------------------"
    # python3 3_fields_over_time.py --folder $1
    # echo "---------------------- 6 ----------------------"
    # python3 6_compare_fields_for_diff.py --folder $1
    echo "---------------------- 7 ----------------------"
    python3 7_stress_difference_components.py --folder $1
}

bigtext() {
    pyfiglet -f starwars -w 140 "$@"
}

COUNTER=0

for file in $@; do
    let COUNTER++
    bigtext $COUNTER
    bigtext $(basename $file);
    run_analysis $file;
done