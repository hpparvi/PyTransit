!!=== Mandel & Agol transit model ===
!!
!! This module implements the uniform and quadratic Mandel & Agol 
!! transit models.
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
!!  28.05.2015

module mandelagol
  use omp_lib
  implicit none

  integer, parameter :: dp=kind(0.d0)
  integer, parameter :: sp=kind(0.0 )

  real(8), parameter :: PI = acos(-1.d0)
  real(8), parameter :: HALF_PI = 0.5d0*PI
  real(8), parameter :: INV_PI = 1.d0/PI
  integer, parameter :: MAXITER = 500
contains

  subroutine eval_uniform(z0,k,c,nthr,nz,flux)
    implicit none
    integer, intent(in) :: nz,nthr
    real(8), intent(in) :: z0(nz), c
    real(8), intent(out) :: flux(nz)
    real(8) :: k,z,lambdae,kap0,kap1
    integer :: i
    
    if(abs(k-0.5d0) < 1.d-3) then
       k=0.5d0
    end if

    !$ call omp_set_num_threads(nthr)
    !$omp parallel do default(none) shared(z0,k,c,nthr,nz,flux) &
    !$omp private(i,z,lambdae,kap0,kap1)
    do i=1,nz
       z=z0(i)
       if(z < 0.d0 .or. z > 1.d0+k) then
          flux(i)=1.d0
          
       else if(k > 1.d0 .and. z < k-1.d0) then
          flux(i)=0.d0
          
       else if(z > abs(1.d0-k) .and. z < 1.d0+k) then
          kap1=acos(min((1.d0-k*k+z*z)/2.d0/z,1.d0))
          kap0=acos(min((k*k+z*z-1.d0)/2.d0/k/z,1.d0))
          lambdae=k*k*kap0+kap1
          lambdae=(lambdae-0.5d0*sqrt(max(4.d0*z*z-(1.d0+z*z-k*k)**2,0.d0)))/pi
          flux(i)=1.d0-lambdae

       else if(z < 1.d0-k) then
          flux(i)=1.d0-k*k
       end if

       flux(i) = c + (1.d0 - c)*flux(i)
    end do
    !$omp end parallel do
  end subroutine eval_uniform


  subroutine eval_quad(z0,k,u,c,nthr,npt,npb,flux)
    implicit none
    integer, intent(in) :: nthr, npt, npb
    real(8), intent(in), dimension(npt) :: z0
    real(8), intent(in) :: u(2*npb), c(npb)
    real(8), intent(out), dimension(npt, npb) :: flux
    real(8) :: k,k2,z2,x1,x2,x3,z,omega(npb),kap0,kap1,q,Kk,Ek,Pk,n,ed,le,ld
    integer :: i,j,iu,iv

    if(abs(k-0.5) < 1.d-4) then 
       k=0.5
    end if

    k2 = k**2
    do i=1,npb
       omega(i) = 1.d0 - u(2*i-1)/3.d0 - u(2*i)/6.d0
    end do

    !$ call omp_set_num_threads(nthr)
    !$omp parallel do default(none) shared(z0,flux,u,c,omega,k,k2,npt,npb) &
    !$omp private(i,j,iu,iv,z,z2,x1,x2,x3,kap0,kap1,q,Kk,Ek,Pk,n,ed,ld,le)
    do i=1,npt
       z=z0(i)

       if (abs(z-k) < 1d-6) then
          z = z+1d-6
       end if

       !! The source is unocculted
       if (z > 1.d0+k .or. z < 0.d0) then
          flux(i,:) = 1.d0
          cycle

       !! The source is completely occulted
       else if (k >= 1.d0 .and. z <= k-1.d0) then
          flux(i,:) = 0.d0
          cycle
       end if

       z2=z**2
       x1=(k-z)**2
       x2=(k+z)**2
       x3=k**2-z**2

       !! The source is partially occulted and the occulting object crosses the limb
       !! Equation (26):
       if(z >= abs(1.d0-k) .and. z <= 1.d0+k) then
          kap1 = acos(min((1.d0-k2+z2)/(2.d0*z),   1.d0))
          kap0 = acos(min((k2+z2-1.d0)/(2.d0*k*z), 1.d0))
          le = k2*kap0+kap1
          le = (le-0.5d0*sqrt(max(4.d0*z2-(1.d0+z2-k2)**2,0.d0)))*INV_PI
       end if

       ! the occulting object transits the source star (but doesn't completely cover it):
       if(z <= 1.0-k) then
          le=k2
       end if

       ! the edge of the occulting star lies at the origin- special expressions in this case:
       if(abs(z-k) < 1.d-4*(z+k)) then
          ! Table 3, Case V.:
          if (k == 0.5d0) then	
             ld = 1.0/3.0 - 4.0*INV_PI/9.0
             ed = 3.0/32.0;
          else if(z > 0.5d0) then
             q=0.50/k
             Kk=ellk(q)
             Ek=ellec(q)
             ld= 1.0/3.0+16.0*k/9.0*INV_PI * (2.0*k2-1.0)*Ek-(32.0*k**4-20.0*k2+3.0)/9.0*INV_PI/k*Kk
             ed= 1.0/2.0*INV_PI * (kap1+k2*(k2+2.0*z2)*kap0-(1.0+5.0*k2+z2)/4.0*sqrt((1.0-x1)*(x2-1.0)))
          else if (z < 0.5d0) then
             ! Table 3, Case VI.:
             q=2.0*k
             Kk=ellk(q)
             Ek=ellec(q)
             ld=1.0/3.0+2.0/9.0*INV_PI*(4.0*(2.0*k2-1.0)*Ek+(1.0-4.0*k2)*Kk)
             ed=k2/2.0*(k2+2.0*z2)
          end if
       end if

       ! the occulting star partly occults the source and crosses the limb:
       ! Table 3, Case III:
       if((z > 0.5+abs(k-0.5) .and. z < 1.0+k).or.(k > 0.50 .and. z > abs(1.0-k) .and. z < k)) then
          q=sqrt((1.0-(k-z)**2)/4.0/z/k)
          Kk=ellk(q)
          Ek=ellec(q)
          n=1.0/x1-1.0
          Pk = ellpicb(n,q) 
          ld= 1.0/9.0*INV_PI/sqrt(k*z) * (((1.0-x2)*(2.0*x2+x1-3.0)-3.0*x3*(x2-2.0))*Kk+4.0*k*z*(z2+7.0*k2-4.0)*Ek-3.0*x3/x1*Pk)
          if(z < k) then
             ld=ld+2.0/3.0
          end if
          ed= 1.0/2.0*INV_PI * (kap1+k2*(k2+2.0*z2)*kap0-(1.0+5.0*k2+z2)/4.0*sqrt((1.0-x1)*(x2-1.0)))
       end if

       ! the occulting star transits the source:
       ! Table 3, Case IV.:
       if(k <= 1.0 .and. z <= (1.0-k)) then
          q=sqrt((x2-x1)/(1.0-x1))
          Kk=ellk(q)
          Ek=ellec(q)
          n=x2/x1-1.0
          Pk = ellpicb(n,q) 
          ld=2.0/9.0*INV_PI/sqrt(1.0-x1)*((1.0-5.0*z2+k2+x3*x3)*Kk+(1.0-x1)*(z2+7.0*k2-4.0)*Ek-3.0*x3/x1*Pk)
          if(z < k) ld=ld+2.0/3.0
          if(abs(k+z-1.0) < 1.d-4) then
             ld= 2.0/3.0*INV_PI*acos(1.0-2.0*k)-4.0/9.0*INV_PI*sqrt(k*(1.0-k))*(3.0+2.0*k-8.0*k2)
          end if
          ed= k2/2.0*(k2+2.0*z2)
       end if

       do j=1,npb
          iu = 2*j-1
          iv = 2*j
          flux(i,j) = 1.d0 - ((1.d0-u(iu)-2.d0*u(iv))*le + (u(iu)+2.d0*u(iv))*ld + u(iv)*ed)/omega(j)
       end do
       flux(i,:) = c + (1.d0-c)*flux(i,:)
    end do
    !$omp end parallel do
  end subroutine eval_quad


  subroutine eval_quad_bilerp(z,k,u,c,nthr,edt,ldt,let,kt,zt,npt,npb,nk,nz,flux)
    implicit none
    integer, intent(in) :: nthr, npt, npb, nk, nz
    real(8), intent(in), dimension(npt) :: z
    real(8), intent(in) :: k, u(2*npb), c(npb), edt(nk,nz), ldt(nk,nz), let(nk,nz), kt(nk), zt(nz)
    real(8), intent(out), dimension(npt, npb) :: flux
    real(8) :: ak, az, dk, dz, ed, le, ld, omega(npb)
    real(8), dimension(2,nz) :: ed2, le2, ld2
    integer :: ik, iz, i, j, iu, iv

    if (k<kt(1) .or. k>kt(nk)) then
       flux = 0.d0
    else
       !$ call omp_set_num_threads(nthr)
       do i=1,npb
          omega(i) = 1.d0 - u(2*i-1)/3.d0 - u(2*i)/6.d0
       end do

       dk = kt(2) - kt(1)
       dz = zt(2) - zt(1)

       ik = floor((k-kt(1))/dk) + 1
       ak = (k-kt(ik))/dk

       !! Copy the k-rows into smaller interpolation arrays for better memory access
       !! (yes, this actually speeds up the code notably)
       ed2 = edt(ik:ik+1,:)
       le2 = let(ik:ik+1,:)
       ld2 = ldt(ik:ik+1,:)

       !$omp parallel do default(none) private(i,j,iz,iu,iv,az,ed,le,ld) &
       !$omp shared(z,zt,dz,u,c,npt,k,flux,ik,ak,edt,let,ldt,npb,omega,ed2,le2,ld2)
       do i=1,npt
          if ((z(i) >= 1.d0+k) .or. (z(i)<0.d0)) then
             flux(i,:) = 1.d0
          else
             iz = floor((z(i)-zt(1))/dz) + 1
             az = (z(i)-zt(iz))/dz

             ed =     ed2(1,iz  )*(1.d0-ak)*(1.d0-az) &
                  & + ed2(2,iz  )*ak*(1.d0-az) &
                  & + ed2(1,iz+1)*(1.d0-ak)*az &
                  & + ed2(2,iz+1)*ak*az

             le =     le2(1,iz  )*(1.d0-ak)*(1.d0-az) &
                  & + le2(2,iz  )*ak*(1.d0-az) &
                  & + le2(1,iz+1)*(1.d0-ak)*az &
                  & + le2(2,iz+1)*ak*az

             ld =     ld2(1,iz  )*(1.d0-ak)*(1.d0-az) &
                  & + ld2(2,iz  )*ak*(1.d0-az) &
                  & + ld2(1,iz+1)*(1.d0-ak)*az &
                  & + ld2(2,iz+1)*ak*az

             do j=1,npb
                iu = 2*j-1
                iv = 2*j
                flux(i,j) = 1.d0 - ((1.d0-u(iu)-2.d0*u(iv))*le + (u(iu)+2.d0*u(iv))*ld + u(iv)*ed)/omega(j)
             end do
             flux(i,:) = c + (1.d0-c)*flux(i,:)
          end if
       end do
       !$omp end parallel do
    end if

  end subroutine eval_quad_bilerp

  subroutine calculate_interpolation_tables(kmin,kmax,nk,nz,nthr,ed,le,ld,kt,zt)
    implicit none
    integer, intent(in) :: nk,nz,nthr
    real(8), intent(in) :: kmin,kmax
    real(8), intent(out), dimension(nk,nz) :: ed,le,ld
    real(8), intent(out) :: zt(nz), kt(nk)
    real(8) :: k,z,k2,z2,x1,x2,x3,kap0,kap1,q,Kk,Ek,Pk,n
    integer :: i,j

    !! FIXME: The code stalls for some combinations of k and z. Find out why and fix.

    zt = [((1+kmax)/real(nz-1,8)*i, i=0,nz-1)]
    kt = [(kmin + (kmax-kmin)/real(nk-1,8)*i, i=0,nk-1)]
    
    !$ call omp_set_num_threads(1)
    do j=1,nk
       k = kt(j)
       k2 = k**2
       !$omp parallel do default(none) shared(kt,zt,k,k2,nz,j,ed,ld,le) &
       !$omp private(z,z2,x1,x2,x3,kap0,kap1,q,Kk,Ek,Pk,n)
       do i=1,nz
          z=zt(i)
          z2=z**2
          x1=(k-z)**2
          x2=(k+z)**2
          x3=k**2-z**2

          if (abs(z-k) < 1d-6) then
             z = z+1d-6
          end if

          if(z < 0.d0 .or. z > 1.d0+k) then
             ld(j,i)=0.d0
             ed(j,i)=0.d0
             le(j,i)=0.d0

          else if(k > 1.d0 .and. z < k-1.d0) then
             ld(j,i)=1.d0
             ed(j,i)=1.d0
             le(j,i)=1.d0

          ! the source is partly occulted and the occulting object crosses the limb:
          ! Equation (26):
          else if(z > abs(1.d0-k) .and. z < 1.d0+k) then
             kap1=acos(min((1.d0-k2+z2)/2.d0/z,1.d0))
             kap0=acos(min((k2+z2-1.d0)/2.d0/k/z,1.d0))
             le(j,i) = k2*kap0+kap1
             le(j,i) = (le(j,i)-0.5d0*sqrt(max(4.d0*z2-(1.d0+z2-k2)**2,0.d0)))*INV_PI
          end if

          ! the occulting object transits the source star (but doesn't completely cover it):
          if(z < 1.d0-k) then
             le(j,i)=k2
          end if

          ! the edge of the occulting star lies at the origin- special expressions in this case:
          if(abs(z-k) < 1.d0-4*(z+k)) then
             ! Table 3, Case V.:
             if(z > 0.5d0) then
                q=0.5d0/k
                Kk=ellk(q)
                Ek=ellec(q)
                ld(j,i)= 1.d0/3.d0+16.d0*k/9.d0*INV_PI * (2.d0*k2-1.d0)*Ek-(32.d0*k**4-20.d0*k2+3.d0)/9.d0*INV_PI/k*Kk
                ed(j,i)= 1.d0/2.d0*INV_PI * (kap1+k2*(k2+2.d0*z2)*kap0-(1.d0+5.d0*k2+z2)/4.d0*sqrt((1.d0-x1)*(x2-1.d0)))
                if(k == 0.5d0) then
                   ld(j,i)=1.d0/3.d0-4.d0*INV_PI/9.d0
                   ed(j,i)=3.d0/32.d0
                end if
             else
                ! Table 3, Case VI.:
                q=2.d0*k
                Kk=ellk(q)
                Ek=ellec(q)
                ld(j,i)=1.d0/3.d0+2.d0/9.d0*INV_PI*(4.d0*(2.d0*k2-1.d0)*Ek+(1.d0-4.d0*k2)*Kk)
                ed(j,i)=k2/2.d0*(k2+2.d0*z2)
             end if
          end if

          ! the occulting star partly occults the source and crosses the limb:
          ! Table 3, Case III:
          if((z > 0.5d0+abs(k-0.5d0) .and. z < 1.d0+k).or.(k > 0.5d0 .and. z > abs(1.d0-k) .and. z < k)) then
             q=sqrt((1.d0-(k-z)**2)/4.d0/z/k)
             Kk=ellk(q)
             Ek=ellec(q)
             n=1.d0/x1-1.d0
             Pk = ellpicb(n,q) 
             ld(j,i)= 1.d0/9.d0*INV_PI/sqrt(k*z) * (((1.d0-x2)*(2.d0*x2+x1-3.d0)-3.d0*x3*(x2-2.d0)) &
                  & *Kk+4.d0*k*z*(z2+7.d0*k2-4.d0)*Ek-3.d0*x3/x1*Pk)
             if(z < k) then
                ld(j,i)=ld(j,i)+2.d0/3.d0
             end if
             ed(j,i)= 1.d0/2.d0*INV_PI * (kap1+k2*(k2+2.d0*z2)*kap0-(1.d0+5.d0*k2+z2)/4.d0*sqrt((1.d0-x1)*(x2-1.d0)))
          end if

          ! the occulting star transits the source:
          ! Table 3, Case IV.:
          if(k < 1.d0 .and. z < (1.d0-k)) then
             q=sqrt((x2-x1)/(1.d0-x1))
             Kk=ellk(q)
             Ek=ellec(q)
             n=x2/x1-1.d0
             Pk = ellpicb(n,q) 

             if(abs(k+z-1.d0) < 1.d-4) then
                ld(j,i) = 2.d0/3.d0*INV_PI*acos(1.d0-2.d0*k)-4.d0/9.d0*INV_PI*sqrt(k*(1.d0-k))*(3.d0+2.d0*k-8.d0*k2)
             else
                ld(j,i) = 2.d0/9.d0*INV_PI/sqrt(1.d0-x1)*((1.d0-5.d0*z2+k2+x3*x3)*Kk+(1.d0-x1)*(z2+7.d0*k2-4.d0)*Ek-3.d0*x3/x1*Pk)
                if(z < k) then
                   ld(j,i) = ld(j,i)+2.d0/3.d0
                end if
             end if
             
             ed(j,i)= k2/2.d0*(k2+2.d0*z2)
          end if
       end do
       !$omp end parallel do
    end do
  end subroutine calculate_interpolation_tables


  real(8) function ellpicb(n, k)
    !! The complete elliptical integral of the third kind
    !!
    !! Bulirsch 1965, Numerische Mathematik, 7, 78
    !! Bulirsch 1965, Numerische Mathematik, 7, 353
    !!
    !! Adapted from L. Kreidbergs C version in BATMAN
    !! (Kreidberg, L. 2015, PASP 957, 127)
    !! (https://github.com/lkreidberg/batman)
    !! which is translated from J. Eastman's IDL routine
    !! in EXOFAST (Eastman et al. 2013, PASP 125, 83)
    !!
    real(8), intent(in) :: n,k
    real(8) :: kc,p,m0,c,d,e,f,g
    integer :: nit

    kc = sqrt(1.d0-k*k)
    p  = sqrt(n + 1.d0)    
    m0 = 1.d0
    c  = 1.d0
    d  = 1.d0/p
    e  = kc

    nit = 0
    do while (nit < 10000)
       f = c
       c = d/p + c
       g = e/p
       d = 2.*(f*g + d)
       p = g + p
       g = m0
       m0 = kc + m0

       if(abs(1.d0-kc/g) > 1.0d-8) then
          kc = 2.d0*sqrt(e)
          e = kc*m0
       else
          ellpicb = HALF_PI*(c*m0+d)/(m0*(m0+p))
          return
       end if
       nit = nit+1
    end do
    ellpicb = 0.d0
  end function ellpicb

  real(8) function ellec(k)
    implicit none
    real(8), intent(in) :: k
    real(8), parameter :: &
         & a1=0.443251414630_dp, &
         & a2=0.062606012200_dp, &
         & a3=0.047573835460_dp, &
         & a4=0.017365064510_dp, &
         & b1=0.249983683100_dp, &
         & b2=0.092001800370_dp, &
         & b3=0.040696975260_dp, &
         & b4=0.005264496390_dp
    real(8) :: m1,ee1,ee2

    m1=1.0-k*k
    ee1=1.0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*log(1.0/m1)
    ellec=ee1+ee2
  end function ellec

  real(8) function ellk(k)
    implicit none
    real(8), intent(in) :: k
    real(8), parameter :: &
         & a0=1.386294361120_dp, &
         & a1=0.096663442590_dp, &
         & a2=0.035900923830_dp, &
         & a3=0.037425637130_dp, &
         & a4=0.014511962120_dp, &
         & b0=0.50_dp, &
         & b1=0.124985935970_dp, &
         & b2=0.068802485760_dp, &
         & b3=0.033283553460_dp, &
         & b4=0.004417870120_dp
    real(8) :: ek1,ek2,m1

    m1=1.0-k*k
    ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*log(m1)
    ellk=ek1-ek2
  end function ellk
end module mandelagol
