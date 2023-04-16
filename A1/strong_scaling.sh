problem_size=8000000

for i in 1 2 4 8; # Processes
do
  echo "Processes: ${i}, Size: ${problem_size}"
  mpirun --bind-to none -n ${i} ./stencil /home/maya/public/PDP_Assignment1/input${problem_size}.txt ./output.txt 100
  echo -e "\n"
done
