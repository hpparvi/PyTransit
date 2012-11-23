!!=== Gimenez transit shape ===
!!
!! This module implements the transit shape model by
!! Alvaro Gimenez (A&A 450, 1231--1237, 2006).
!!
!! The code is adapted from the Fortran implementation
!! at http://thor.ieec.uab.es/LRVCode. The major changes
!! are in the parallelization and vectorization. 
!!
!! Instead of repeating the computations for each lightcurve point
!! separately, we minimize the redundant calculations by
!! computing the common factors for all lightcurve points.
!! This can give us a speed up of several orders of magnitude. 
!!
!! -GPL-
!! This program is free software: you can redistribute it and/or modify
!! it under the terms of the GNU General Public License as published by
!! the Free Software Foundation, either version 3 of the License, or
!! (at your option) any later version.
!!
!! This program is distributed in the hope that it will be useful,
!! but WITHOUT ANY WARRANTY; without even the implied warranty of
!! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!! GNU General Public License for more details.
!!
!! You should have received a copy of the GNU General Public License
!! along with this program.  If not, see <http://www.gnu.org/licenses/>.
!! -GPL-
!!
!! Author
!!  Hannu Parviainen <hannu@iac.es>
!!
!! Date 
!!  01.03.2011
!!

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 128
#endif

module gimenez
  use, intrinsic :: ISO_C_BINDING
  implicit none

  integer, parameter :: FS = C_FLOAT
  integer, parameter :: FD = C_DOUBLE

  real(8), parameter :: PI = 3.1415926525_fd
  real(8), parameter :: HALF_PI = 0.5_fd * PI

  real(8), allocatable, dimension(:,:) :: a_e_nm, a_e_vl
  real(8), allocatable, dimension(:,:,:) :: j_d, j_e
  real(8), allocatable, dimension(:) :: tmp1d1, tmp1d2

contains
  subroutine init(npol, nldc)
    implicit none
    integer, intent(in) :: npol, nldc
    integer :: i, j
    real(8) :: nu, nm1

    if (allocated(a_e_nm)) deallocate(a_e_nm)
    if (allocated(a_e_vl)) deallocate(a_e_vl)
    if (allocated(j_d))    deallocate(j_d)
    if (allocated(j_e))    deallocate(j_e)
    if (allocated(tmp1d1)) deallocate(tmp1d1)
    if (allocated(tmp1d2)) deallocate(tmp1d2)

    allocate(a_e_nm(npol, nldc+1), a_e_vl(npol, nldc+1))
    allocate(j_d(4, npol, nldc+1), j_e(4, npol, nldc+1))
    allocate(tmp1d1(1000000), tmp1d2(1000000))

    do j=1,nldc+1
       nu = (real(j-1,8)+2._fd)/2._fd
       do i = 0, npol-1
          nm1 = exp(log_gamma(nu+i+1._fd) - log_gamma(i+2._fd))
          a_e_vl(i+1, j) = (-1)**(i) * (2._fd+2._fd*i+nu) * nm1
          a_e_nm(i+1, j) = exp(log_gamma(i+1._fd) + log_gamma(nu+1._fd) - log_gamma(i+1._fd+nu))
       end do
    end do

    !! Jacobi coefficients
    !!
    do j=1,nldc+1
       nu = (real(j-1,8)+2._fd)/2._fd
       do i = 0, npol-1
          call j_coeff(0._fd, 1._fd+nu, real(i+1,8), j_d(:,i+1,j))
          call j_coeff(   nu,    1._fd, real(i+1,8), j_e(:,i+1,j))
       end do
    end do
  
  contains
    subroutine j_coeff(alpha, beta, ri, res)
      implicit none
      real(8), intent(in) :: alpha, beta, ri
      real(8), dimension(4), intent(inout):: res
      res(1) = 2._fd * ri * ( ri + alpha + beta ) * ( 2._fd * ri - 2._fd + alpha + beta )
      res(2) = ( 2._fd* ri - 1._fd + alpha + beta ) * ( 2._fd * ri  + alpha + beta ) * ( 2._fd* ri - 2._fd + alpha + beta )
      res(3) = ( 2._fd* ri - 1._fd + alpha + beta ) * ( alpha + beta ) * ( alpha - beta )
      res(4) = - 2._fd * ( ri - 1._fd + alpha ) * ( ri - 1._fd + beta )  * ( 2._fd* ri + alpha + beta )
    end subroutine j_coeff
  end subroutine init

  subroutine finalize()
    implicit none
    if (allocated(a_e_nm)) deallocate(a_e_nm)
    if (allocated(a_e_vl)) deallocate(a_e_vl)
    if (allocated(j_d)) deallocate(j_d)
    if (allocated(j_e)) deallocate(j_e)
  end subroutine finalize


  subroutine eval_lerp(z, r, b, u, contamination, npol, nthreads, update, npt, nldc, res)
    use omp_lib
    implicit none
    integer, intent(in) :: npt, nldc, nthreads, npol
    logical, intent(in) :: update
    real(8), intent(in), dimension(npt) :: z
    real(8), intent(in), dimension(nldc) :: u
    real(8), intent(in) :: r, b, contamination
    real(8), intent(out), dimension(npt) :: res

    integer, parameter :: tsize = 512
    real(8), save, dimension(tsize) :: z_tbl, i_tbl
    real(8), save :: dz, idz 
    real(8) :: x
    integer :: i, k, ntr

    logical, dimension(npt) :: mask
    real(8), dimension(npt) :: ztmp, itmp

    !$ if (nthreads /= 0) call omp_set_num_threads(nthreads)
 
    if (update) then
       dz    = (1._fd+r - b)/real(tsize-1, fd)
       idz   = 1._fd/dz
       z_tbl = [(b + i*dz, i=0,tsize-1)]
       call eval(z_tbl, r, u, contamination, npol, nthreads, tsize, nldc, i_tbl)
    end if

    mask        = (z > 0._fd) .and. (z < 1._fd+r)
    ntr         = count(mask)
    ztmp(1:ntr) = pack(z, mask)

    !$omp parallel do private(i, x, k) shared(npt, z, ntr, r, b, i_tbl, idz, ztmp, itmp) default(none)
    do i=1,ntr
       x = (ztmp(i)-b)*idz
       k = 1 + int(floor(x))
       x = x - k
       k = min(k, tsize-1)
       itmp(i) = (1._fd-x) * i_tbl(k) + x * i_tbl(k+1)
    end do 
    !$omp end parallel do
 
    res = 1._fd
    res = unpack(itmp(1:ntr), mask, res)
  end subroutine eval_lerp
 

  subroutine eval(z, r, u, contamination, npol, nthreads, npt, nldc, res)
    use omp_lib
    implicit none
    integer, intent(in) :: npt, nldc, nthreads, npol
    real(8), intent(in), dimension(npt) :: z
    real(8), intent(in), dimension(nldc) :: u
    real(8), intent(in) :: r, contamination
    real(8), intent(out), dimension(npt) :: res
    
    logical, dimension(npt) :: mask
    integer :: npt_t
    !real(8) :: t_start, t_finish

    !$ if (nthreads /= 0) call omp_set_num_threads(nthreads)
    res  = 0._fd
    mask = (z > 0._fd) .and. (z < 1._fd+r)
    npt_t = count(mask)

    tmp1d1(1:npt_t) = pack(z, mask)
    tmp1d2(1:npt_t) = gimenez_v(tmp1d1(1:npt_t), u, r, npol)

    res = unpack(tmp1d2(1:npt_t), mask, res)
    res = 1._fd + res*(1._fd - contamination)
  end subroutine eval


  function gimenez_v(z, u, r, npol)
    implicit none
    real(8), intent(in), dimension(:) :: z
    real(8), intent(in), dimension(:) :: u    
    real(8), intent(in) :: r
    integer, intent(in) :: npol

    real(8), dimension(size(z)) :: gimenez_v

    real(8), dimension(size(z), size(u)+1) :: a 
    real(8), dimension(size(u)+1) :: n, Cn
    real(8), dimension(1) :: b
    real(8), dimension(size(z)) :: c
    real(8), dimension(size(u)) :: uu ! Change to quadratic
    integer :: i, j 

    a  = 0._fd
    Cn = 1._fd
    n  = [(i, i=0,size(u))]
    b  = r/(1._fd+r)
    c  = z/(1._fd+r)
    !TODO: FIXME, I'M A HACK
    if (size(u) == 2) then
       uu = [u(1)+2._fd*u(2), -u(2)]
    else
       uu = [u(1), 0.0d0]
    end if

    do j=0,size(u)
       call alpha(b, c, j, npol, a(:,j+1))
    end do

    if (size(u) > 0) then
       Cn(1) = (1._fd - sum(uu)) / (1._fd - sum(uu*n(2:) / (n(2:)+2._fd)))
       Cn(2:) = uu / (1._fd - sum(n(2:) * uu / (n(2:)+2._fd)))
    end if

    !$omp parallel do private(i), shared(z,gimenez_v, a, Cn) default(none)
    do i=1,size(z)
       gimenez_v(i) = -sum(a(i,:)*Cn)
    end do
    !$omp end parallel do
  end function gimenez_v
  
  !!--- Alpha ---
  !!
  subroutine alpha(b, c, n, npol, a)
    implicit none
    real(8), intent(in), dimension(:) :: b, c
    integer, intent(in) :: n, npol
    real(8), dimension(size(c)), intent(out) :: a

    real(8), dimension(size(c)) :: norm, sm
    real(8), dimension(1, npol) :: e
    real(8), dimension(CHUNK_SIZE, npol) :: d

    real(8) :: nu
    integer :: i, n_chunks, i_chunk, ic_s, ic_e

    n_chunks = size(c) / CHUNK_SIZE
    nu   = (real(n,8)+2._fd)/2._fd
    sm = 0._fd

    !$omp parallel private(i, i_chunk, ic_s, ic_e, d) default(shared)
    !$omp workshare
    norm = b(1)*b(1) * (1._fd - c*c)**(1._fd + nu) / (nu * gamma(1._fd + nu))
    !$omp end workshare

    !$omp single
    call jacobi(npol,  nu,  1._fd,  1._fd-2._fd*(1._fd-b),  n+1,  j_e,  e)
    do i = 1, npol
       e(1,i) = e(1,i) * a_e_nm(i,n+1)
       e(1,i) = e(1,i)**2
    end do
    !$omp end single

    !$omp do
    do i_chunk = 0, n_chunks-1
       ic_s = 1+(i_chunk)*CHUNK_SIZE
       ic_e = (i_chunk+1)*CHUNK_SIZE
       call jacobi(npol, 0._fd, 1._fd+nu, 1._fd-2._fd*c(ic_s:ic_e)**2, n+1, j_d, d)
       do i = 1, npol
          sm(ic_s:ic_e)  = sm(ic_s:ic_e) + (a_e_vl(i,n+1) * d(:,i) * e(1,i))
       end do
    end do
    !$omp end do

    !$omp single
    ic_s = n_chunks*CHUNK_SIZE + 1
    call jacobi(npol, 0._fd, 1._fd+nu, 1._fd-2._fd*c(ic_s:)**2, n+1, j_d, d(:size(c)-ic_s+1,:))
    do i = 1, npol
       sm(ic_s:)  = sm(ic_s:) + (a_e_vl(i,n+1) * d(:size(c)-ic_s+1,i) * e(1,i))
    end do
    !$omp end single

    !$omp workshare
    a = norm * sm
    !$omp end workshare
    !$omp end parallel

end subroutine alpha

  !!--- Jacobi polynomials ---
  !!
  !! Adapted from the Jacobi polynomials routine by J. Burkardt. The
  !! only major difference is that the routine computes the values for
  !! multiple x at the same time.
  !!
  subroutine jacobi(npol, alpha, beta, x, i_ld, j_c, cx)
    use omp_lib
    implicit none
    integer, intent(in) :: npol, i_ld
    real(8), intent(in) :: alpha, beta
    real(8), intent(in), dimension(:) :: x
    real(8), intent(in), dimension(:,:,:) :: j_c
    real(8), intent(out), dimension(size(x),npol) :: cx

    integer :: i, j, k
    j = i_ld
    
    cx(:,1) = 1._fd
    cx(:,2) = ( 1._fd + 0.5_fd * ( alpha + beta ) ) * x + 0.5_fd * ( alpha - beta )

    do i = 2, npol-1
       do k = 1, size(x)
          cx(k, i+1) = ( ( j_c(3,i,j) + j_c(2,i,j) * x(k) ) * cx(k,i) + j_c(4,i,j) * cx(k,i-1) ) / j_c(1,i,j)
       end do
    end do
  end subroutine jacobi

end module gimenez
