# Compile with gcc and all warnings
gcc -Wall -o average_parallel average_parallel.c -lpthread

# Define array of matrix dimensions
declare -a dim=(  73 179 283 419 547 661 811 947 1087 1229 1381 4073 )
# Define array of number of threads
declare -a threads=( 2 3 4 5 6 7 8 )

# Time the program for each dimension and output to file
for i in "${dim[@]}"
do
    for j in "${threads[@]}"
    do
        echo "Running for dimension $i with $j threads"
        time ./average_parallel $i 0.1 $j
    done
done &> output.txt