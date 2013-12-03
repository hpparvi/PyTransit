!!=== Gimenez transit shape ===
!!
!! This module implements the transit shape model by
!! Alvaro Gimenez (A&A 450, 1231--1237, 2006).
!!
!! The code is adapted from the original Fortran implementation
!! at http://thor.ieec.uab.es/LRVCode. The major changes
!! are in the precalculation of coefficients and in separation
!! of the limb darkening effects from rest of the computations.
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
!!  Hannu Parviainen <parviainen@astro.ox.ac.uk>
!!
!! Date 
!!  01.03.2011
!!

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 128
#endif

module gimenez
  use, intrinsic :: ISO_C_BINDING
  use omp_lib
  implicit none

  integer, parameter :: FS = C_FLOAT
  integer, parameter :: FD = C_DOUBLE

  real(8), parameter :: PI = 3.1415926525_fd
  real(8), parameter :: HALF_PI = 0.5_fd * PI

contains
  subroutine init_arrays(npol, nldc, anm, avl, ajd, aje)
    implicit none
    integer, intent(in) :: npol, nldc
    real(8), intent(out), dimension(npol, nldc+1) :: anm, avl
    real(8), intent(out), dimension(4, npol, nldc+1) :: ajd, aje

    real(8) :: nu, nm1
    integer :: i, j

    do j=1,nldc+1
       nu = (real(j-1,8)+2._fd)/2._fd
       do i = 0, npol-1
          nm1 = exp(log_gamma(nu+i+1._fd) - log_gamma(i+2._fd))
          avl(i+1, j) = (-1)**(i) * (2._fd+2._fd*i+nu) * nm1
          anm(i+1, j) = exp(log_gamma(i+1._fd) + log_gamma(nu+1._fd) - log_gamma(i+1._fd+nu))
       end do
    end do

    !! Jacobi coefficients
    !!
    do j=1,nldc+1
       nu = (real(j-1,8)+2._fd)/2._fd
       do i = 0, npol-1
          call j_coeff(0._fd, 1._fd+nu, real(i+1,8), ajd(:,i+1,j))
          call j_coeff(   nu,    1._fd, real(i+1,8), aje(:,i+1,j))
       end do
    end do
  
  contains
    subroutine j_coeff(alpha, beta, ri, res)
      implicit none
      real(fd), intent(in) :: alpha, beta, ri
      real(fd), dimension(4), intent(inout):: res
      res(1) = 2._fd * ri * ( ri + alpha + beta ) * ( 2._fd * ri - 2._fd + alpha + beta )
      res(2) = ( 2._fd* ri - 1._fd + alpha + beta ) * ( 2._fd * ri  + alpha + beta ) * ( 2._fd* ri - 2._fd + alpha + beta )
      res(3) = ( 2._fd* ri - 1._fd + alpha + beta ) * ( alpha + beta ) * ( alpha - beta )
      res(4) = - 2._fd * ( ri - 1._fd + alpha ) * ( ri - 1._fd + beta )  * ( 2._fd* ri + alpha + beta )
    end subroutine j_coeff
  end subroutine init_arrays


  subroutine eval_lerp(z, k, u, b, contamination, nthreads, update, npol, npt, npb, nldc, anm, avl, ajd, aje, res)
    implicit none
    integer, intent(in) :: npt, nldc, npb, nthreads, npol
    logical, intent(in) :: update
    real(8), intent(in), dimension(npt) :: z
    real(8), intent(in), dimension(nldc, npb) :: u
    real(8), intent(in) :: k, b, contamination
    real(8), intent(out), dimension(npt, npb) :: res
    real(8), intent(in), dimension(npol, nldc+1) :: anm, avl
    real(8), intent(in), dimension(4, npol, nldc+1) :: ajd, aje

    integer, parameter :: tsize =  32
    integer, parameter :: maxpb =  20

    real(8) :: dz_ft, dz_ie, idz_ft, idz_ie
    real(8), dimension(tsize) :: ft_ztable, ie_ztable
    real(8), dimension(tsize,maxpb) :: ft_table, ie_table

    real(8) :: x
    integer :: i, j, ntr

    logical, dimension(npt) :: mask
    real(8), dimension(npt) :: ztmp
    real(8), dimension(npt, npb) :: itmp

     !$ if (nthreads /= 0) call omp_set_num_threads(nthreads)
 
    dz_ft = (1._fd-k)/real(tsize-1,8)
    dz_ie = 2*k/real(tsize-1,8)
    idz_ft = 1._fd / dz_ft
    idz_ie = 1._fd / dz_ie

    ft_ztable = [(0._fd   + dz_ft*i, i=0,tsize-1)]
    ie_ztable = [(1._fd-k + dz_ie*i, i=0,tsize-1)]

    ft_table(:,:npb) = 1._fd + gimenez_m(ft_ztable, k, u, npol, nldc, npb, anm, avl, ajd, aje) * (1._fd - contamination)
    ie_table(:,:npb) = 1._fd + gimenez_m(ie_ztable, k, u, npol, nldc, npb, anm, avl, ajd, aje) * (1._fd - contamination)
    ie_table(tsize,:npb) = 1._fd

    mask        = (z > 0._fd) .and. (z < 1._fd+k)
    ntr         = count(mask)
    ztmp(1:ntr) = pack(z, mask)
    
    !$omp parallel do private(i, x, j) shared(ntr, k, b, idz_ft, idz_ie, ft_ztable) &
    !$omp shared(ie_ztable, ft_table, ie_table, npb, dz_ft, ztmp, itmp) default(none)
    do i=1,ntr
       if (ztmp(i) <= 1._fd-k) then
          x = ztmp(i)*idz_ft
          j = 1 + int(floor(x))
          x = x - j + 1
          j = min(j, size(ft_ztable)-1)
          itmp(i,:) = (1._fd-x) * ft_table(j, :npb) + x * ft_table(j+1, :npb)
       else
          x = (ztmp(i)-(1._fd-k))*idz_ie
          j = 1 + int(floor(x))
          x = x - j + 1
          j = min(j, size(ie_table)-1)
          itmp(i,:) = (1._fd-x) * ie_table(j, :npb) + x * ie_table(j+1, :npb)
       end if
    end do 
    !$omp end parallel do
    
    res = 1._fd
    do i=1,npb
       res(:,i) = unpack(itmp(1:ntr,i), mask, res(:,i))
    end do
  end subroutine eval_lerp
 

  subroutine eval(z, k, u, contamination, nthreads, npol, npt, nldc, npb, anm, avl, ajd, aje, res)
    implicit none
    integer, intent(in) :: npt, nldc, npb, nthreads, npol
    real(8), intent(in), dimension(npt) :: z
    real(8), intent(in), dimension(nldc, npb) :: u
    real(8), intent(in) :: k, contamination
    real(8), intent(out), dimension(npt, npb) :: res
    real(8), intent(in), dimension(npol, nldc+1) :: anm, avl
    real(8), intent(in), dimension(4, npol, nldc+1) :: ajd, aje

    real(8), dimension(npt) :: tmp1
    real(8), dimension(npt,npb) :: tmp2
    logical, dimension(npt) :: mask
    integer :: npt_t, i

    !$ if (nthreads /= 0) call omp_set_num_threads(nthreads)
    res  = 0._fd
    mask = (z > 0._fd) .and. (z < 1._fd+k)
    npt_t = count(mask)
    
    tmp1(1:npt_t) = pack(z, mask)
    tmp2(1:npt_t,:) = gimenez_m(tmp1(1:npt_t), k, u, npol, nldc, npb, anm, avl, ajd, aje)

    do i=1,npb
       res(:,i) = unpack(tmp2(1:npt_t,i), mask, res(:,i))
    end do
    res = 1._fd + res*(1._fd - contamination)
  end subroutine eval

  !!--- Gimenez transit shape model ---
  !!
  function gimenez_m(z, k, u, npol, nldc, npb, anm, avl, ajd, aje)
    implicit none
    real(8), intent(in), dimension(:) :: z
    real(8), intent(in), dimension(nldc,npb) :: u    
    real(8), intent(in) :: k
    integer, intent(in) :: npol, nldc, npb
    real(8), intent(in), dimension(npol, nldc+1) :: anm, avl
    real(8), intent(in), dimension(4, npol, nldc+1) :: ajd, aje

    real(8), dimension(size(z), npb) :: gimenez_m

    real(8), dimension(size(z), size(u,1)+1) :: a 
    real(8), dimension(size(u,1)+1) :: n
    real(8), dimension(size(u,1)+1,npb) :: Cn

    real(8), dimension(1) :: b
    real(8), dimension(size(z)) :: c
    real(8), dimension(size(u,1), npb) :: uu ! Quadratic -> general
    integer :: i, j, ui 

    a  = 0._fd
    Cn = 1._fd
    n  = [(i, i=0,size(u,1))]
    b  = k/(1._fd+k)
    c  = z/(1._fd+k)

    if (size(u,1) == 2) then
       uu(1,:) =  u(1,:) + 2._fd*u(2,:)
       uu(2,:) = -u(2,:)
    else
       uu = u
    end if

    do j=0,size(u,1)
       call alpha(b, c, j, npol, nldc, a(:,j+1), anm, avl, ajd, aje)
    end do

    if (size(u,1) > 0) then
       do ui=1,npb
          Cn(1,ui)  = (1._fd - sum(uu(:,ui))) / (1._fd - sum(uu(:,ui)*n(2:) / (n(2:)+2._fd)))
          Cn(2:,ui) = uu(:,ui) / (1._fd - sum(n(2:) * uu(:,ui) / (n(2:)+2._fd)))
       end do
    end if
    
    !$omp parallel do private(i,j), shared(z,gimenez_m, a, Cn, npb) default(none)
    do i=1,size(z)
       do j=1,npb
          gimenez_m(i,j) = -sum(a(i,:)*Cn(:,j),1)
       end do
    end do
    !$omp end parallel do
  end function gimenez_m
  

  !!--- Alpha ---
  !!
  subroutine alpha(b, c, n, npol, nldc, a, anm, avl, ajd, aje)
    implicit none
    real(8), intent(in), dimension(:) :: b, c
    integer, intent(in) :: n, npol, nldc
    real(8), dimension(size(c)), intent(out) :: a
    real(8), intent(in), dimension(npol, nldc+1) :: anm, avl
    real(8), intent(in), dimension(4, npol, nldc+1) :: ajd, aje

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
    call jacobi(npol,  nu,  1._fd,  1._fd-2._fd*(1._fd-b),  n+1,  aje,  e)
    do i = 1, npol
       e(1,i) = e(1,i) * anm(i,n+1)
       e(1,i) = e(1,i)**2
    end do
    !$omp end single

    !$omp do
    do i_chunk = 0, n_chunks-1
       ic_s = 1+(i_chunk)*CHUNK_SIZE
       ic_e = (i_chunk+1)*CHUNK_SIZE
       call jacobi(npol, 0._fd, 1._fd+nu, 1._fd-2._fd*c(ic_s:ic_e)**2, n+1, ajd, d)
       do i = 1, npol
          sm(ic_s:ic_e)  = sm(ic_s:ic_e) + (avl(i,n+1) * d(:,i) * e(1,i))
       end do
    end do
    !$omp end do

    !$omp single
    ic_s = n_chunks*CHUNK_SIZE + 1
    call jacobi(npol, 0._fd, 1._fd+nu, 1._fd-2._fd*c(ic_s:)**2, n+1, ajd, d(:size(c)-ic_s+1,:))
    do i = 1, npol
       sm(ic_s:)  = sm(ic_s:) + (avl(i,n+1) * d(:size(c)-ic_s+1,i) * e(1,i))
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
