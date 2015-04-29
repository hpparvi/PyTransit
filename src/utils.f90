module utils
  use omp_lib
  implicit none

contains

  subroutine average_samples_1(fin, npt, nss, nthreads, fout)
    integer, intent(in) :: npt, nss, nthreads
    real(8), intent(in), dimension(0:npt*nss-1) :: fin
    real(8), intent(out), dimension(0:npt-1) :: fout
    integer :: i
    fout = 0.d0

    !$ if (nthreads /= 0) call omp_set_num_threads(nthreads)
    !$omp parallel do private(i) shared(fin,fout,nss)
    do i=0,npt-1
       fout(i) = sum(fin(i*nss:(i+1)*nss-1))/real(nss,8)
    end do
    !$omp end parallel do
  end subroutine average_samples_1

end module utils
