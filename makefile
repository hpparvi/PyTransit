gcc:
	#export FFLAGS='-Ofast -cpp -fopenmp -fPIC -march=native -DTSTI=100'
	f2py -c -m gimenez gimenez.f90 -lm -lgomp --fcompiler=gnu95 --opt='-Ofast -ffast-math' --f90flags='-cpp -fopenmp -march=native -DCHUNK_SIZE=128'

intel:
	export FFLAGS='-O3 -openmp -parallel -par-report3 -openmp-report3'
	f2py --fcompiler=intelem -c -m gimenez gimenez.f90

