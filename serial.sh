# Compile with gcc and all warnings
gcc -Wall -o average_serial average_serial.c

# Define array of matrix dimensions
declare -a dim=( 8 16 32 64 128 256 512 1024 2048 4096 8192 )


# Time the program for each dimension and output to file
for i in "${dim[@]}"
do
    echo "Running for dimension $i"
    time ./average_serial $i 0.1
done &> serial.txt