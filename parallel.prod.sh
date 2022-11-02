# Set TIMEFORMAT
TIMEFORMAT=%R

# Compile with gcc and all warnings
# gcc -Wall -o average_parallel average_parallel.c -lpthread

# Define array of matrix dimensions
declare -a dim=( 8 16 32 64 128 256 512 1024 2048 4096 8192 )
# Define array of number of threads
declare -a threads=( 2 4 8 16 32 44 )

# Time the program for each dimension and output
for j in "${threads[@]}"
do
    for i in "${dim[@]}"
    do
        echo "Running for dimension $i with $j threads"
        time ./average_parallel $i 0.1 $j
    done
done