!!=== Orbits ===
!!
!! This module implements subroutines to compute projected distances, and other
!! utilities for orbit calculations. 
!!
!! - Mean anomaly
!!
!! - Eccentric anomaly
!!   - Newton (two methods)
!!
!! - True anomaly
!!   - Iteration
!!   - Newton
!!   - Series expansion (3)
!!   - Series expansion (5)
!!   - Interpolation
!!
!! - projected distance
!!   - Circular orbit
!!   - Iteration
!!   - Newton
!!   - Series expansion (3)
!!   - Series expansion (5)
!!   - Interpolation
!!
!! - Utility functions
!!   - Impact parameter for a circular orbit
!!   - Impact parameter for an eccentric orbit
!!   - Transit duration for a circular orbit
!!   - Transit duration for an eccentric orbit
!!   - Eclipse shift in time relative to p/2
!!
!! -GPL-
!!
!! Copyright (C) 2010--2015  Hannu Parviainen
!!
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
!!  Hannu Parviainen <hannu.parviainen@physics.ox.ac.uk>
!!
!! Date 
!!  13.04.2012

module orbits
  use, intrinsic :: ISO_C_BINDING
  use omp_lib
  implicit none

  integer, parameter :: FS = C_FLOAT
  integer, parameter :: FD = C_DOUBLE

  real(fd), parameter ::      PI = 3.1415926525_fd
  real(fd), parameter ::  INV_PI = 1.0_fd / PI
  real(fd), parameter ::  TWO_PI = 2.0_fd * PI
  real(fd), parameter :: HALF_PI = 0.5_fd * PI

contains

  integer pure function iclip(v, vmin, vmax)
    implicit none
    integer, intent(in) :: v, vmin, vmax
    iclip = min(max(v, vmin), vmax)
  end function iclip

  real(8) pure function rclip(v, vmin, vmax)
    implicit none
    real(8), intent(in) :: v, vmin, vmax
    rclip = min(max(v, vmin), vmax)
  end function rclip

  
  !! ============
  !! MEAN ANOMALY
  !! ============
  
  !! Calculates the time offset between the zero mean anomaly and transit
  !! center knowing that the true anomaly f is f(t_tr) = pi/2 - w.
  real(fd) function mean_anomaly_offset(e, w)
    implicit none
    real(fd), intent(in) :: e, w

    mean_anomaly_offset = atan2(sqrt(1._fd-e**2)*sin(half_pi - w), e + cos(half_pi - w))
    mean_anomaly_offset = mean_anomaly_offset - e*sin(mean_anomaly_offset)
  end function mean_anomaly_offset

  
  subroutine mean_anomaly(t, t0, p, e, w, nth, nt, Ma)
    implicit none
    integer, intent(in) :: nt, nth
    real(fd), intent(in) :: t0, p, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: Ma
    real(fd) :: offset

    offset = mean_anomaly_offset(e, w)
    Ma = modulo(two_pi * (t - (t0 - offset*p/two_pi))/ p, two_pi)
  end subroutine mean_anomaly

  
  !! ==================
  !! PROJECTED DISTANCE
  !! ==================
  !! Methods to calculate the projected normalized planet-star distance for a planet in an eccentric orbit.
  !! Parameters
  !!   t    d[nt]  time                          [d]
  !!   t0   d      mid-transit time              [d]
  !!   p    d      period                        [d]
  !!   a    d      scaled semi-major axis        [Rs]
  !!   i    d      inclination                   [rad]
  !!   e    d      eccentricity                  [-]
  !!   w    d      argument of the periastron    [rad]
  !!   nth  i      number of openmp threads
  !!   nt   i      number of points

  subroutine z_eccentric_from_ta(Ta, a, i, e, w, nth, nt, z)
    implicit none
    integer, intent(in) :: nt, nth
    real(fd), intent(in), dimension(nt) :: Ta
    real(fd), intent(in) :: a, i, e, w
    real(fd), intent(out), dimension(nt) :: z
    !$ if (nth /= 0) call omp_set_num_threads(nth)
    !$omp parallel workshare
    z = a*(1-e**2)/(1+e*cos(Ta)) * sqrt(1.0_fd - sin(w+Ta)**2 * sin(i)**2) * sign(1._fd, sin(w+Ta))
    !$omp end parallel workshare
  end subroutine z_eccentric_from_ta
  
  subroutine z_circular(t, t0, p, a, i, nth, nt, z)
    implicit none
    integer, intent(in) :: nt, nth
    real(fd), intent(in) :: t0, p, a, i
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: z
    real(fd) :: n, cosph, sini
    integer :: j
    !$ if (nth /= 0) call omp_set_num_threads(nth)
    sini = sin(i)
    n    = TWO_PI/p
    !$omp parallel do private(j, cosph) shared(nt, a, t, t0, sini, n, z) default(none)
    do j=1,nt
       cosph = cos((t(j)-t0)*n)
       z(j) = sign(1._fd, cosph)*a*sqrt(1._fd - cosph*cosph*sini*sini)
    end do
    !$omp end parallel do
  end subroutine z_circular
  
  subroutine z_eccentric_iter(t, t0, p, a, i, e, w, nth, nt, z)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, a, i, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: z
    real(fd), dimension(nt) :: Ta 

    call ta_eccentric_iter(t, t0, p, e, w, nth, nt, Ta)
    call z_eccentric_from_ta(Ta, a, i, e, w, nth, nt, z)
  end subroutine z_eccentric_iter

  subroutine z_eccentric_newton(t, t0, p, a, i, e, w, nth, nt, z)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, a, i, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: z
    real(fd), dimension(nt) :: Ta 

    call ta_eccentric_newton(t, t0, p, e, w, nth, nt, Ta)
    call z_eccentric_from_ta(Ta, a, i, e, w, nth, nt, z)
  end subroutine z_eccentric_newton

  subroutine z_eccentric_ps3(t, t0, p, a, i, e, w, nth, nt, z)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, a, i, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: z
    real(fd), dimension(nt) :: Ta

    call ta_eccentric_ps3(t, t0, p, e, w, nth, nt, Ta)
    call z_eccentric_from_ta(Ta, a, i, e, w, nth, nt, z)
  end subroutine z_eccentric_ps3

  subroutine z_eccentric_ps5(t, t0, p, a, i, e, w, nth, nt, z)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, a, i, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: z
    real(fd), dimension(nt) :: Ta

    call ta_eccentric_ps5(t, t0, p, e, w, nth, nt, Ta)
    call z_eccentric_from_ta(Ta, a, i, e, w, nth, nt, z)
  end subroutine z_eccentric_ps5

  subroutine z_eccentric_ip(t, t0, p, a, i, e, w, nth, nt, z)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, a, i, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: z
    real(fd), dimension(nt) :: Ta

    call ta_eccentric_bilerp(t, t0, p, e, w, nth, nt, Ta)
    call z_eccentric_from_ta(Ta, a, i, e, w, nth, nt, z)
  end subroutine z_eccentric_ip

  !! =================
  !! ECCENTRIC ANOMALY
  !! =================

  subroutine ea_eccentric_newton(t, t0, p, e, w, nth, nt, Ea)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: Ea
    
    real(fd), dimension(:), allocatable :: Ma  ! Mean anomaly

    integer j, k
    real(fd) :: m_offset, err

    allocate(Ma(nt))
    call mean_anomaly(t, t0, p, e, w, nth, nt, Ma)

    !$ if (nth /= 0) call omp_set_num_threads(nth)
    !$omp parallel private(j,k,err) shared(nt,t,t0,m_offset,p,e,Ma,Ea) default(none)
    Ea = Ma
    !$omp do schedule(guided)
    do j = 1, nt
       err = 0.05_fd
       k = 0
       do while ((abs(err) > 1.0d-8) .and. (k<1000)) 
          err   = Ea(j) - e*sin(Ea(j)) - Ma(j)
          Ea(j) = Ea(j) - err/(1._fd-e*cos(Ea(j)))
          k = k + 1
       end do
    end do
    !$omp end do
    !$omp end parallel
    
    deallocate(Ma)
  end subroutine ea_eccentric_newton

 subroutine ea_eccentric_newton2(t, m0, p, e, w, nth, nt, Ea)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: m0, p, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: Ea
    real(fd), dimension(:), allocatable :: Ma
    integer j
    real(fd) :: err

    allocate(Ma(nt))

    !$ if (nth /= 0) call omp_set_num_threads(nth)
    !$omp parallel private(j,err) shared(nt,t,m0,p,e,Ma,Ea) default(none)

    !! Calculate the mean anomaly
    !$omp workshare
    Ma = two_pi*t/p + m0
    !$omp end workshare

    !! Calculate the eccentric anomaly using the Newton's method
    Ea = Ma
    !$omp do schedule(guided)
    do j = 1, nt
       err = 0.05_fd
       do while (abs(err) > 1.0d-8) 
          err   = Ea(j) - e*sin(Ea(j)) - Ma(j)
          Ea(j) = Ea(j) - err/(1._fd-e*cos(Ea(j)))
       end do
    end do
    !$omp end do
    !$omp end parallel
    
    deallocate(Ma)
  end subroutine ea_eccentric_newton2

  !! ============
  !! TRUE ANOMALY
  !! ============

  !! TA - CIRCULAR ORBIT
  !! ===================
  subroutine ta_circular(t, t0, p, nth, nt, ta)
    !! Calculates the true anomaly for a planet in a circular orbit.

    implicit none
    integer, intent(in) :: nt, nth
    real(fd), intent(in) :: t0, p
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: ta
    integer :: j

    !$ if (nth /= 0) call omp_set_num_threads(nth)
    !$omp parallel do private(j) shared(nt, t, p, t0, ta) default(none)
    do j=1,nt
       ta(j) = (t(j)-t0)*(TWO_PI/p) + HALF_PI
    end do
    !$omp end parallel do
  end subroutine ta_circular 


  !! TA - ECCENTRIC ORBIT
  !! ====================
  !! Different methods to calculate the true anomaly for a planet in an eccentric orbit.
  !! Parameterization is the same for all the methods.
  !!
  !! Parameters
  !!   t    d[nt]  time                          [d]
  !!   t0   d      mid-transit time              [d]
  !!   p    d      period                        [d]
  !!   e    d      eccentricity                  [-]
  !!   w    d      argument of the periastron    [rad]
  !!   nth  i      number of openmp threads
  !!   nt   i      number of points

  subroutine ta_from_ma(Ma, e, nth, nt, Ta)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: e
    real(fd), intent(in),  dimension(nt) :: Ma
    real(fd), intent(out), dimension(nt) :: Ta
    real(fd) :: Ea, ec, ect, cta, sta
    integer :: j, k
    
    !$ if (nth /= 0) call omp_set_num_threads(nth)
    !$omp parallel private(j,k,Ea,ec,ect,cta,sta) shared(nt,e,Ma,Ta) default(none)
    !$omp do schedule(guided)
    do j = 1, nt
       ec = e*sin(Ma(j))/(1._fd - e*cos(Ma(j)))
       do k = 1, 15          
          ect = ec
          ec  = e*sin(Ma(j)+ec)
          if (abs(ect-ec) < 1d-4) exit
       end do
       Ea = Ma(j) + ec
       sta = sqrt(1-e**2) * sin(Ea)/(1.0_fd-e*cos(Ea))
       cta = (cos(Ea)-e)/(1.0_fd-e*cos(Ea))
       Ta(j) = atan2(sta, cta) 
    end do
    !$omp end do
    !$omp end parallel
  end subroutine ta_from_ma


  subroutine ta_from_ma_2(Ma, e, nth, nt, Ta)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: e
    real(fd), intent(in),  dimension(nt) :: Ma
    real(fd), intent(out), dimension(nt) :: Ta
    real(fd), dimension(:), allocatable :: Ea, ec, cta, sta
    integer :: j, k
    real(fd) :: ect
    allocate(Ea(nt), ec(nt), cta(nt), sta(nt))
    
    !$ if (nth /= 0) call omp_set_num_threads(nth)
    !$omp parallel private(j,k,ect) shared(ec,nt,e,Ma,Ea,sta,cta,Ta) default(none)

    ec = e*sin(Ma)/(1._fd - e*cos(Ma))

    !! Calculate the eccentric anomaly using iteration
    !$omp do schedule(guided)
    do j = 1, nt
       do k = 1, 15
          ect   = ec(j)
          ec(j) = e*sin(Ma(j)+ec(j))
          if (abs(ect-ec(j)) < 1d-4) exit
       end do
    end do
    !$omp end do

    !$omp workshare
    Ea  = Ma + ec
    sta = sqrt(1-e**2) * sin(Ea)/(1-e*cos(Ea))
    cta = (cos(Ea)-e)/(1-e*cos(Ea))
    Ta  = atan2(sta, cta) 
    !$omp end workshare
    !$omp end parallel
    deallocate(Ea, cta, sta)
  end subroutine ta_from_ma_2

  
  subroutine ta_eccentric_iter(t, t0, p, e, w, nth, nt, Ta)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: Ta
    real(fd), dimension(:), allocatable :: Ma, Ea, ec, cta, sta
    integer :: j, k
    real(fd) :: m_offset, ect
    allocate(Ma(nt), Ea(nt), ec(nt), cta(nt), sta(nt))

    call mean_anomaly(t, t0, p, e, w, nth, nt, Ma)
    
    !$ if (nth /= 0) call omp_set_num_threads(nth)
    !$omp parallel private(j,k,ect) shared(ec,nt,t,t0,m_offset,p,e,Ma,Ea,sta,cta,Ta) default(none)

    ec = e*sin(Ma)/(1._fd - e*cos(Ma))

    !! Calculate the eccentric anomaly using iteration
    !$omp do schedule(guided)
    do j = 1, nt
       do k = 1, 15
          ect   = ec(j)
          ec(j) = e*sin(Ma(j)+ec(j))
          if (abs(ect-ec(j)) < 1d-4) exit
       end do
    end do
    !$omp end do

    !! Calculate the true anomaly from the eccentric anomaly
    !$omp workshare
    Ea  = Ma + ec
    sta = sqrt(1-e**2) * sin(Ea)/(1-e*cos(Ea))
    cta = (cos(Ea)-e)/(1-e*cos(Ea))
    Ta  = atan2(sta, cta) 
    !$omp end workshare
    !$omp end parallel
    deallocate(Ma, Ea, cta, sta)
  end subroutine ta_eccentric_iter

  
  subroutine ta_eccentric_newton(t, t0, p, e, w, nth, nt, Ta)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: Ta
    real(fd), dimension(:), allocatable :: Ea, cta, sta
    allocate(Ea(nt), cta(nt), sta(nt))
    
    call ea_eccentric_newton(t, t0, p, e, w, nth, nt, Ea)
    
    !! Calculate the true anomaly from the eccentric anomaly
    ! $omp workshare
    sta = sqrt(1-e**2) * sin(Ea)/(1-e*cos(Ea))
    cta = (cos(Ea)-e)/(1-e*cos(Ea))
    Ta  = atan2(sta, cta)
    ! $omp end workshare
    ! $omp end parallel
    deallocate(Ea, cta, sta)
  end subroutine ta_eccentric_newton

  
  subroutine ta_eccentric_ps3(t, t0, p, e, w, nth, nt, Ta)
    !! Calculates the true anomaly using a series expansion.
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: Ta
    real(fd), dimension(:), allocatable :: Ma  ! Mean anomaly
    real(fd) :: m_offset

    allocate(Ma(nt))
    call mean_anomaly(t, t0, p, e, w, nth, nt, Ma)
    !$ if (nth /= 0) call omp_set_num_threads(nth)
    !$omp parallel workshare default(shared)
    Ta = Ma + (2._fd*e - 0.25_fd*e**3)*sin(Ma) &
         &  + 1.25_fd*e**2*sin(2*Ma) &
         &  + 13._fd/12._fd*e**3*sin(3*Ma)
    !$omp end parallel workshare
    deallocate(Ma)
  end subroutine ta_eccentric_ps3


  subroutine ta_eccentric_ps5(t, t0, p, e, w, nth, nt, Ta)
    !! Calculate the true anomaly using a seriest expansion from Meeus (p. 222). 
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: Ta
    real(fd), dimension(:), allocatable :: Ma  ! Mean anomaly
    real(fd) :: m_offset

    allocate(Ma(nt))
    call mean_anomaly(t, t0, p, e, w, nth, nt, Ma)
    !$ if (nth /= 0) call omp_set_num_threads(nth)
    !$omp parallel workshare default(shared)
    Ta = Ma + (2._fd*e - 0.25_fd*e**3 + 5._fd/96._fd*e**5) * sin(Ma) &
         &  + (1.25_fd*e**2 - 11._fd/24._fd*e**4) * sin(2*Ma) &
         &  + (13._fd/12._fd * e**3 - 43._fd/64._fd * e**5) * sin(3*Ma) &
         &  + 103._fd/96._fd * e**4 * sin(4*Ma) &
         &  + 1097._fd/960._fd * e**5 * sin(5*Ma)
    !$omp end parallel workshare
    deallocate(Ma)
  end subroutine ta_eccentric_ps5


  !! Calculates the eccentric anomaly for time values `t` given
  !! eccentricity `e` using bilinear interpolation.
  !!
  !! Parameters
  !!
  !! t  Time array
  !! e  Eccentricity
  subroutine ta_eccentric_bilerp(t, t0, p, e, w, nth, npt, Ta)
    implicit none
    integer,  parameter :: NM = 250
    integer,  parameter :: NE = 150
    integer,  intent(in) :: nth, npt
    real(fd), intent(in) :: t0, p, e, w
    real(fd), intent(in) :: t(npt)
    real(fd), intent(out) :: Ta(npt)
    real(fd) :: tbl2(2,NM), mi(NM), ae, am, s, dm, de
    integer :: i, j, ie, im

    real(8), dimension(:,:), allocatable, save :: tbl
    real(8), dimension(:), allocatable :: Ma
    allocate(Ma(npt))
    
    de = 0.95_fd / real(NE-1, 8)
    dm = PI / real(NM-1, 8)

    if (.not. allocated(tbl)) then
       mi = (/ ((j-1)*dm, j=1,nm) /)
       allocate(tbl(ne,nm))
       do i=1,ne
          call ta_from_ma(mi, (i-1)*de, 1, NM, tbl(i,:))
          tbl(i,:) = tbl(i,:) - mi
       end do
    end if
    
    ie = iclip(     floor(e/de) + 1,      1,   NE-1)
    ae = rclip((e - de*(ie-1)) / de, 0.0_fd, 1.0_fd)
    tbl2 = tbl(ie:ie+1,:)

    call mean_anomaly(t, t0, p, e, w, nth, npt, Ma)

    !$ if (nth /= 0) call omp_set_num_threads(nth)
    !$omp parallel do private(i,im,am,s) shared(Ma,npt,dm,tbl2,ae)
    do i=1,npt
       if (Ma(i) < PI) then
          im = iclip(floor(Ma(i)/dm) + 1, 1, NM-1)
          am = rclip((Ma(i) - dm*(im-1)) / dm, 0.0_fd, 1.0_fd)
          s = 1.0_fd
       else
          im = iclip(floor((TWO_PI - Ma(i))/dm) + 1, 1, NM-1)
          am = rclip((TWO_PI - (Ma(i) - dm*(im-1))) / dm, 0.0_fd, 1.0_fd)
          s = -1.0_fd
       end if
          
       Ta(i) =   tbl2(1,im  )*(1d0-ae)*(1d0-am) &
             & + tbl2(2,im  )*     ae *(1d0-am) &
             & + tbl2(1,im+1)*(1d0-ae)*     am  &
             & + tbl2(2,im+1)*     ae *     am
       Ta(i) = Ma(i) + s * Ta(i)

       if (Ta(i) < 0.0_fd) then
          Ta(i) = Ta(i) + TWO_PI
       end if
    end do
    !$omp end parallel do
    deallocate(Ma)
  end subroutine ta_eccentric_bilerp

  subroutine ta_eccentric_ip(t, t0, p, e, w, nth, update, nt, ta)
    !! Calculates the true anomaly using linear interpolation.
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: ta
    logical, intent(in) :: update

    integer, parameter :: tsize = 512
    real(fd), dimension(tsize), save :: t_table, ta_table
    real(fd), save :: dt, idt
    real(fd) ::x
    integer j, k
    logical :: sflag
    
    !! FIXME: This routine is unaccurate for large values of t 

    !$ if (nth /= 0) call omp_set_num_threads(nth)

    sflag = .false.
    if(update) then
       dt  = p/real(tsize-1, fd)
       idt = 1._fd/dt
       t_table = [(dt*j, j=0,tsize-1)]
       call ta_eccentric_iter(t_table, t0, p, e, w, nth, tsize, ta_table)
       do j=2,tsize
          if (.not. sflag) then
             if (ta_table(j-1) > ta_table(j)) sflag = .true.
          end if
          if (sflag) then
             ta_table(j) = ta_table(j) + 2*PI
          end if
       end do
    end if

    !$omp parallel do private(j,x,k) shared(nt,t,p,idt,ta_table,ta) default(none) schedule(static)
    do j=1,nt
       x = mod(t(j), p)
       if (x<0._fd) x = p+x
       x = x*idt
       k = int(floor(x)) + 1
       x = x - k + 1
       ta(j) = (1._fd-x) * ta_table(k) + x * ta_table(k+1)
    end do
    !$omp end parallel do
  end subroutine ta_eccentric_ip

  !! ================
  !! UTILITY ROUTINES
  !! ================
  
  !! Approximate shift of the minima (quite good for e < 0.5)
  !! Kallrath, J., "Eclipsing Binary Stars: Modeling and Analysis", p. 85
  real(fd) function eclipse_shift_ap(p, i, e, w)
    implicit none
    real(fd), intent(in) :: p, i, e, w
    eclipse_shift_ap = 0.5_fd*p + (p*inv_pi * e * cos(w) * (1._fd + 1._fd / sin(i)**2))
  end function eclipse_shift_ap

  !! Exact shift of the minima, good for a eccentricities.
  real(fd) function eclipse_shift_ex(p, i, e, w)
    real(fd), intent(in) :: p, i, e, w
    real(fd) :: etr, eec, mtr, mec

    etr = atan2(sqrt(1._fd-e**2)*sin(half_pi - w), e + cos(half_pi - w))
    eec = atan2(sqrt(1._fd-e**2)*sin(half_pi + pi - w), e + cos(half_pi + pi - w))
    mtr = etr - e*sin(etr)
    mec = eec - e*sin(eec)

    eclipse_shift_ex = (mec-mtr)*p/two_pi

    if (eclipse_shift_ex < 0._fd) eclipse_shift_ex = p + eclipse_shift_ex
  end function eclipse_shift_ex

  !! Transit duration for a circular orbit
  !! Seager & Mallen-Ornelas (2003)
  real(fd) elemental function duration_circular(p, a, i)
    implicit none
    real(fd), intent(in) :: p, a, i
    duration_circular = p*inv_pi * asin(sqrt(1._fd - a**2 *cos(i)**2)/(a*sin(i)))
  end function duration_circular
  

  !! Transit duration T14 for an eccentric orbit
  !! Winn, J.
  !!
  !! Note: This is slightly more accurate than the duration by Ford et al.
  real(fd) elemental function duration_eccentric_w(p, k, a, i, e, w, tr_sign)
    implicit none
    real(fd), intent(in) :: p, k, a, i, e, w, tr_sign
    real(fd) :: ae !! Eq. (16) in Winn, J. 
    real(fd) :: b

    b  = impact_parameter_ec(a, i, e, w, tr_sign)
    ae = sqrt(1._fd-e**2)/(1+tr_sign*e*sin(w))
    duration_eccentric_w = p*inv_pi  * asin(sqrt((1._fd+k)**2-b**2)/(a*sin(i))) * ae
  end function duration_eccentric_w

  !! Transit duration T14 for an eccentric orbit
  !! Ford et al. (2008)
  real(fd) elemental function duration_eccentric_f(p, k, a, i, e, w, tr_sign)
    implicit none
    real(fd), intent(in) :: p, k, a, i, e, w, tr_sign
    real(fd) :: ae !! ae = d_t/a, as presented near Eq. (1) in Ford et al. (2008).
    real(fd) :: b

    b  = impact_parameter_ec(a, i, e, w-half_pi, tr_sign)
    ae = (1._fd-e**2)/(1+tr_sign*e*sin(w))
    duration_eccentric_f = p*inv_pi  * 1._fd/(a*sqrt(1._fd-e**2)) * sqrt((1+k)**2 -b**2) * ae
  end function duration_eccentric_f

  !! Transit duration from center crossing to center crossing for an eccentric orbit
  !! Tingley & Sackett (2005)
  real(fd) function duration_eccentric_ts(p, a, i, e, w)
    implicit none
    real(fd), intent(in) :: p, a, i, e, w
    real(fd) :: z
    duration_eccentric_ts = 0.0_fd
  end function duration_eccentric_ts

  !! Impact parameter
  !! ================

  !! Circular orbit
  !! --------------
  !!
  real(fd) elemental function impact_parameter(a, i)
    real(fd), intent(in) :: a, i
    impact_parameter = a * cos(i)
  end function impact_parameter

  !! Eccentric orbit
  !! ---------------------------
  !! tr_sign: transit sign, +1 for transits, -1 for eclipses
  !! Winn, J., "Transits and Occultations", 2010
  !!
  real(fd) elemental function impact_parameter_ec(a, i, e, w, tr_sign)
    real(fd), intent(in) :: a, i, e, w, tr_sign
    impact_parameter_ec = a * cos(i) * ((1-e**2) / (1+tr_sign*e*sin(w)))
  end function impact_parameter_ec

end module orbits
