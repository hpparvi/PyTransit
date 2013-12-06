!!=== Orbits ===
!!
!! This module implements subroutines to compute projected distances, and other
!! utilities for orbit calculations. 
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
!!  Hannu Parviainen <parviainen@astro.ox.ax.uk>
!!
!! Date 
!!  13.04.2012
!!
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
  subroutine z_circular(t, t0, p, a, i, nthreads, nt, z)
    implicit none
    integer, intent(in) :: nt, nthreads
    real(fd), intent(in) :: t0, p, a, i
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: z

    real(fd) :: n, cosph, sini
    integer :: j

    !$ if (nthreads /= 0) call omp_set_num_threads(nthreads)

    sini = sin(i)
    n    = TWO_PI/p
    !$omp parallel do private(j, cosph) shared(nt, a, t, t0, sini, n, z) default(none)
    do j=1,nt
       cosph = cos((t(j)-t0)*n)
       z(j) = sign(1._fd, cosph)*a*sqrt(1._fd - cosph*cosph*sini*sini)
    end do
    !$omp end parallel do
  end subroutine z_circular

  !! z_circular_mt - calculate z for a circular orbit and multiple transit center times
  !!
  subroutine z_circular_mc(t, t0, t0npt, p, a, i, nthreads, nt, nt0, z)
    implicit none
    integer, intent(in) :: nt, nt0, nthreads
    real(fd), intent(in) :: p, a, i
    real(fd), intent(in),  dimension(nt0) :: t0
    integer,  intent(in),  dimension(nt0) :: t0npt
    real(fd), intent(in),  dimension(nt)  :: t
    real(fd), intent(out), dimension(nt)  :: z

    integer, dimension(nt0) :: t0start
    real(fd) :: n, cosph, sini
    integer :: j,k, jloc

    t0start(1) = 0
    do j=2,nt0
       t0start(j) = t0start(j-1) + t0npt(j-1)
    end do

    sini = sin(i)
    n    = TWO_PI/p

    !$ if (nthreads /= 0) call omp_set_num_threads(nthreads)
    !$omp parallel do private(k, j, jloc, cosph) shared(a, t, t0, nt0, t0start, t0npt, sini, n, z) default(none)
    do k=1,nt0
       do j=1,t0npt(k)
          jloc = t0start(k)+j
          cosph = cos((t(jloc)-t0(k))*n)
          z(jloc) = sign(1._fd, cosph)*a*sqrt(1._fd - cosph*cosph*sini*sini)
       end do
    end do
    !$omp end parallel do
  end subroutine z_circular_mc


  subroutine z_eccentric_from_ta(Ta, a, i, e, w, nth, nt, z)
    implicit none
    integer, intent(in) :: nt, nth
    real(fd), intent(in), dimension(nt) :: Ta
    real(fd), intent(in) :: a, i, e, w
    real(fd), intent(out), dimension(nt) :: z
    integer :: j
    !$ if (nth /= 0) call omp_set_num_threads(nth)
    !$omp parallel do private(j) shared(z,a,i,e,w,Ta,nt) default(none) schedule(static)
    do j=1,nt
       z(j) = a*(1-e**2)/(1+e*cos(Ta(j))) * sqrt(1 - sin(w+Ta(j))**2 * sin(i)**2)
    end do
    !$omp end parallel do
  end subroutine z_eccentric_from_ta


  !! ==================
  !! PROJECTED DISTANCE
  !! ==================
  !! Different methods to calculate the projected distance for a planet in an eccentric orbit.
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

  subroutine z_eccentric(t, t0, p, a, i, e, w, nth, nt, z)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, a, i, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: z
    real(fd), dimension(nt) :: Ta 

    call ta_eccentric(t, t0, p, e, w, nth, nt, Ta)
    call z_eccentric_from_ta(Ta, a, i, e, w, nth, nt, z)
  end subroutine z_eccentric

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

  subroutine z_eccentric_s3(t, t0, p, a, i, e, w, nth, nt, z)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, a, i, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: z
    real(fd), dimension(nt) :: Ta

    call ta_eccentric_s3(t, t0, p, e, w, nth, nt, Ta)
    call z_eccentric_from_ta(Ta, a, i, e, w, nth, nt, z)
  end subroutine z_eccentric_s3

  subroutine z_eccentric_s(t, t0, p, a, i, e, w, nth, nt, z)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, a, i, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: z
    real(fd), dimension(nt) :: Ta

    call ta_eccentric_series(t, t0, p, e, w, nth, nt, Ta)
    call z_eccentric_from_ta(Ta, a, i, e, w, nth, nt, z)
  end subroutine z_eccentric_s

  subroutine z_eccentric_ip(t, t0, p, a, i, e, w, nth, update, nt, z)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, a, i, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: z
    real(fd), dimension(nt) :: Ta
    logical, intent(in) :: update

    call ta_eccentric_ip(t, t0, p, e, w, nth, update, nt, Ta)
    call z_eccentric_from_ta(Ta, a, i, e, w, nth, nt, z)
  end subroutine z_eccentric_ip



  !! ============
  !! TRUE ANOMALY
  !! ============

  !! TA - CIRCULAR ORBIT
  !! ===================
  subroutine ta_circular(t, t0, p, nthreads, nt, ta)
    !! Calculates the true anomaly for a planet in a circular orbit.

    implicit none
    integer, intent(in) :: nt, nthreads
    real(fd), intent(in) :: t0, p
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: ta
    integer :: j

    !$ if (nthreads /= 0) call omp_set_num_threads(nthreads)
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
  subroutine ta_eccentric(t, t0, p, e, w, nth, nt, Ta)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: Ta
    
    real(fd), dimension(nt) :: Ma  ! Mean anomaly
    real(fd), dimension(nt) :: Ea  ! Eccentric anomaly
    real(fd), dimension(nt) :: ec  ! Ea = Ma + ec
    real(fd), dimension(nt) :: cta, sta  ! cos(Ta), sin(Ta)
    
    integer j, k
    real(fd) :: m_offset, ect

    !$ if (nth /= 0) call omp_set_num_threads(nth)

    !! Calculate the time offset between the zero mean anomaly and transit
    !! center knowing that t_tr = f_tr = pi/2 - w.
    m_offset = atan2(sqrt(1._fd-e**2)*sin(half_pi - w), e + cos(half_pi - w))
    m_offset = m_offset - e*sin(m_offset)

    !$omp parallel private(j,k,ect) shared(ec,nt,t,t0,m_offset,p,e,Ma,Ea,sta,cta,Ta) default(none)

    !! Calculate the mean anomaly
    !$omp workshare
    Ma = two_pi * (t - (t0 - m_offset*p/two_pi))/ p
    ec = e*sin(Ma)/(1._fd - e*cos(Ma))
    !$omp end workshare

    !! Calculate the eccentric anomaly using iteration
    !$omp do schedule(guided)
    do j = 1, nt
       do k = 1, 15
          ect   = ec(j)
          ec(j) = e*sin(Ma(j)+ec(j))
          if (abs(ect-ec(j)) < 1e-4_fd) exit
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
  end subroutine ta_eccentric

  subroutine ta_eccentric_newton(t, t0, p, e, w, nth, nt, Ta)
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: Ta
    
    real(fd), dimension(nt) :: Ma  ! Mean anomaly
    real(fd), dimension(nt) :: Ea  ! Eccentric anomaly
    real(fd), dimension(nt) :: cta, sta  ! cos(Ta), sin(Ta)
    
    integer j
    real(fd) :: m_offset, err

    !$ if (nth /= 0) call omp_set_num_threads(nth)

    !! Calculate the time offset between the zero mean anomaly and transit
    !! center knowing that t_tr = f_tr = pi/2 - w.
    m_offset = atan2(sqrt(1._fd-e**2)*sin(half_pi - w), e + cos(half_pi - w))
    m_offset = m_offset - e*sin(m_offset)

    !$omp parallel private(j,err) shared(nt,t,t0,m_offset,p,e,Ma,Ea,sta,cta,Ta) default(none)

    !! Calculate the mean anomaly
    !$omp workshare
    Ma = two_pi * (t - (t0 - m_offset*p/two_pi))/ p
    !$omp end workshare

    !! Calculate the eccentric anomaly using the Newton's method
    Ea = Ma
    !$omp do schedule(guided)
    do j = 1, nt
       err = 0.05_fd
       do while (abs(err) > 1.0e-8) 
          err   = Ea(j) - e*sin(Ea(j)) - Ma(j)
          Ea(j) = Ea(j) - err/(1._fd-e*cos(Ea(j)))
       end do
    end do
    !$omp end do

    !! Calculate the true anomaly from the eccentric anomaly
    !$omp workshare
    sta = sqrt(1-e**2) * sin(Ea)/(1-e*cos(Ea))
    cta = (cos(Ea)-e)/(1-e*cos(Ea))
    Ta  = atan2(sta, cta)
    !$omp end workshare
    !$omp end parallel
  end subroutine ta_eccentric_newton

  subroutine ta_eccentric_s3(t, t0, p, e, w, nth, nt, Ta)
    !! Calculates the true anomaly using a series expansion.
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: Ta
    
    real(fd), dimension(nt) :: Ma  ! Mean anomaly
    real(fd) :: m_offset

    !$ if (nth /= 0) call omp_set_num_threads(nth)

    !! Calculate the time offset between the zero mean anomaly and transit
    !! center knowing that t_tr = f_tr = pi/2 - w.
    m_offset = atan2(sqrt(1._fd-e**2)*sin(half_pi - w), e + cos(half_pi - w))
    m_offset = m_offset - e*sin(m_offset)

    !! Calculate the mean anomaly
    !$omp parallel workshare
    Ma = two_pi * (t - (t0 - m_offset*p/two_pi))/ p

    Ta = Ma + (2._fd*e - 0.25_fd*e**3)*sin(Ma) &
         &  + 1.25_fd*e**2*sin(2*Ma) &
         &  + 13._fd/12._fd*e**3*sin(3*Ma)
    !$omp end parallel workshare

  end subroutine ta_eccentric_s3


  subroutine ta_eccentric_series(t, t0, p, e, w, nth, nt, Ta)
    !! Calculate the true anomaly using a seriest expansion from Meeus (p. 222). 
    implicit none
    integer, intent(in)  :: nt, nth
    real(fd), intent(in)  :: t0, p, e, w
    real(fd), intent(in),  dimension(nt) :: t
    real(fd), intent(out), dimension(nt) :: Ta
    
    real(fd), dimension(nt) :: Ma  ! Mean anomaly
    real(fd) :: m_offset

    !$ if (nth /= 0) call omp_set_num_threads(nth)

    !! Calculate the time offset between the zero mean anomaly and transit
    !! center knowing that t_tr = f_tr = pi/2 - w.
    m_offset = atan2(sqrt(1._fd-e**2)*sin(half_pi - w), e + cos(half_pi - w))
    m_offset = m_offset - e*sin(m_offset)

    !$omp parallel workshare default(shared)
    Ma = two_pi * (t - (t0 - m_offset*p/two_pi))/ p

    Ta = Ma + (2._fd*e - 0.25_fd*e**3 + 5._fd/96._fd*e**5) * sin(Ma) &
         &  + (1.25_fd*e**2 - 11._fd/24._fd*e**44) * sin(2*Ma) &
         &  + (13._fd/12._fd * e**3 - 43._fd/64._fd * e**5) * sin(3*Ma) &
         &  + 103._fd/96._fd * e**4 * sin(4*Ma) &
         &  + 1097._fd/960._fd * e**5 * sin(5*Ma)
    !$omp end parallel workshare
  end subroutine ta_eccentric_series


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
    real(fd) ::x, ta_diff
    integer j, k

    !! FIXME: This routine is unaccurate for large t (JDs)!! 

    !$ if (nth /= 0) call omp_set_num_threads(nth)

    if(update) then
       dt  = p/real(tsize-1, fd)
       idt = 1._fd/dt
       t_table = [(dt*j, j=0,tsize-1)]
       call ta_eccentric(t_table, t0, p, e, w, nth, tsize, ta_table)
    end if

    !$omp parallel do private(j,x,k, ta_diff) shared(nt, t, p, idt, ta_table, ta) default(none) schedule(static)
    do j=1,nt
       x = mod(t(j), p)*idt
       k = int(floor(x)) + 1
       x = x - k + 1

       ta_diff = ta_table(k+1) - ta_table(k)

       if (ta_diff >= 0._fd) then
          ta(j) = (1._fd-x) * ta_table(k) + x * ta_table(k+1)
       else
          ta(j) = (1._fd-x) * ta_table(k) + x * PI
       end if

    end do
    !$omp end parallel do
  end subroutine ta_eccentric_ip



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
