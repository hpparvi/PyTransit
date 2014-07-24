module ma_quad
  use omp_lib
  implicit none

contains
  subroutine ma(z0,u1,u2,k,flux,nz)
    implicit none
    integer, intent(in) :: nz
    real(8), intent(in), dimension(nz) :: z0
    real(8), intent(in) :: u1, u2
    real(8), intent(out), dimension(nz) :: flux
    real(8), dimension(nz) ::  mu, lambdad, etad, lambdae
    real(8) :: k,k2,z2,lam, pi,x1,x2,x3,z,omega,kap0,kap1,q,Kk,Ek,Pk,n
    integer :: i

    if(abs(k-0.5) < 1.d-3) then 
       k=0.5
    end if

    ! Input:
    ! z0   impact parameter in units of rs
    ! k    star-planet radius ratio
    ! u1   linear    limb-darkening coefficient (gamma_1 in paper)
    ! u2   quadratic limb-darkening coefficient (gamma_2 in paper)

    ! Limb darkening has the form:
    !  I(r)=[1-u1*(1-sqrt(1-(r/rs)^2))-u2*(1-sqrt(1-(r/rs)^2))^2]/(1-u1/3-u2/6)/pi

    k2 = k**2
    omega=1.-u1/3.d0-u2/6.d0
    pi=acos(-1.d0)

    do i=1,nz
       z=z0(i)
       z2 = z**2
       x1=(k-z)**2
       x2=(k+z)**2
       x3=k**2-z**2

       ! the source is unocculted:
       ! Table 3, I.
       if(z > 1.d0+k) then
          lambdad(i)=0.d0
          etad(i)=0.d0
          lambdae(i)=0.d0
       end if

       ! the  source is completely occulted:
       ! Table 3, II.
       if(k > 1.d0 .and. z < k-1.d0) then
          lambdad(i)=1.d0
          etad(i)=1.d0
          lambdae(i)=1.d0
       end if

       ! the source is partly occulted and the occulting object crosses the limb:
       ! Equation (26):
       if(z > abs(1.d0-k) .and. z < 1.d0+k) then
          kap1=acos(min((1.d0-k2+z2)/2.d0/z,1.d0))
          kap0=acos(min((k2+z2-1.d0)/2.d0/k/z,1.d0))
          lambdae(i) = k2*kap0+kap1
          lambdae(i) = (lambdae(i)-0.5d0*sqrt(max(4.d0*z2-(1.d0+z2-k2)**2,0.d0)))/pi
       end if

       ! the occulting object transits the source star (but doesn't
       ! completely cover it):
       if(z < 1.d0-k) lambdae(i)=k2
       ! the edge of the occulting star lies at the origin- special 
       ! expressions in this case:
       if(abs(z-k) < 1.d-4*(z+k)) then
          ! Table 3, Case V.:
          if(z > 0.5d0) then
             lam=0.5d0*pi
             q=0.5d0/k
             Kk=ellk(q)
             Ek=ellec(q)
             ! Equation 34: lambda_3
             lambdad(i)=1.d0/3.d0+16.d0*k/9.d0/pi*(2.d0*k2-1.d0)*Ek- &
                  &     (32.d0*k**4-20.d0*k2+3.d0)/9.d0/pi/k*Kk
             ! Equation 34: eta_1
             etad(i)=1.d0/2.d0/pi*(kap1+k2*(k2+2.d0*z2)*kap0- &
                  &  (1.d0+5.d0*k2+z2)/4.d0*sqrt((1.d0-x1)*(x2-1.d0)))
             if(k.eq.0.5d0) then
                ! Case VIII: k=1/2, z=1/2
                lambdad(i)=1.d0/3.d0-4.d0/pi/9.d0
                etad(i)=3.d0/32.d0
             end if
          else
             ! Table 3, Case VI.:
             lam=0.5d0*pi
             q=2.d0*k
             Kk=ellk(q)
             Ek=ellec(q)
             ! Equation 34: lambda_4
             lambdad(i)=1.d0/3.d0+2.d0/9.d0/pi*(4.d0*(2.d0*k2-1.d0)*Ek+(1.d0-4.d0*k2)*Kk)
             ! Equation 34: eta_2
             etad(i)=k2/2.d0*(k2+2.d0*z2)
          end if
       end if
       ! the occulting star partly occults the source and crosses the limb:
       ! Table 3, Case III:
       if((z > 0.5d0+abs(k-0.5d0) .and. z < 1.d0+k).or.(k > 0.5d0 .and. z > abs(1.d0-k)*1.0001d0 .and. z < k)) then
          lam=0.5d0*pi
          q=sqrt((1.d0-(k-z)**2)/4.d0/z/k)
          Kk=ellk(q)
          Ek=ellec(q)
          n=1.d0/x1-1.d0
          Pk=Kk-n/3.d0*rj(0.d0,1.d0-q*q,1.d0,1.d0+n)
          ! Equation 34, lambda_1:
          lambdad(i)=1.d0/9.d0/pi/sqrt(k*z)*(((1.d0-x2)*(2.d0*x2+ &
          &        x1-3.d0)-3.d0*x3*(x2-2.d0))*Kk+4.d0*k*z*(z2+ &
          &        7.d0*k2-4.d0)*Ek-3.d0*x3/x1*Pk)
          if(z < k) lambdad(i)=lambdad(i)+2.d0/3.d0
          ! Equation 34, eta_1:
          etad(i)=1.d0/2.d0/pi*(kap1+k2*(k2+2.d0*z2)*kap0- &
          &          (1.d0+5.d0*k2+z2)/4.d0*sqrt((1.d0-x1)*(x2-1.d0)))
       end if
       ! the occulting star transits the source:
       ! Table 3, Case IV.:
       if(k < 1.d0 .and. z < (1.d0-k)*1.0001d0) then
          lam=0.5d0*pi
          q=sqrt((x2-x1)/(1.d0-x1))
          Kk=ellk(q)
          Ek=ellec(q)
          n=x2/x1-1.d0
          Pk=Kk-n/3.d0*rj(0.d0,1.d0-q*q,1.d0,1.d0+n)
          ! Equation 34, lambda_2:
          lambdad(i)=2.d0/9.d0/pi/sqrt(1.d0-x1)*((1.d0-5.d0*z2+k2+&
          &         x3*x3)*Kk+(1.d0-x1)*(z2+7.d0*k2-4.d0)*Ek-3.d0*x3/x1*Pk)
          if(z < k) lambdad(i)=lambdad(i)+2.d0/3.d0
          if(abs(k+z-1.d0) < 1.d-4) then
             lambdad(i)=2/3.d0/pi*acos(1.d0-2.d0*k)-4.d0/9.d0/pi*&
             &            sqrt(k*(1.d0-k))*(3.d0+2.d0*k-8.d0*k2)
          end if
          ! Equation 34, eta_2:
          etad(i)=k2/2.d0*(k2+2.d0*z2)
       end if
       flux(i)=1.d0-((1.d0-u1-2.d0*u2)*lambdae(i)+(u1+2.d0*u2)*lambdad(i)+u2*etad(i))/omega
    end do
  end subroutine ma

  real(8) function rc(x,y)
    real(8), intent(in) :: x,y
    real(8), parameter :: ERRTOL=0.04d0, THIRD=1.d0/3.d0, C1=0.3d0, C2=1.d0/7.d0, C3=0.375d0, C4=9.d0/22.d0
    real(8) :: alamb,ave,s,w,xt,yt

    if(y > 0.d0) then
       xt=x
       yt=y
       w=1.
    else
       xt=x-y
       yt=-y
       w=sqrt(x)/sqrt(xt)
    end if

    s = 1.d3
    do while(abs(s) > ERRTOL)
       alamb=2.d0*sqrt(xt)*sqrt(yt)+yt
       xt=0.25d0*(xt+alamb)
       yt=0.25d0*(yt+alamb)
       ave=THIRD*(xt+yt+yt)
       s=(yt-ave)/ave
    end do

    rc=w*(1.d0+s*s*(C1+s*(C2+s*(C3+s*C4))))/sqrt(ave)
  end function rc
  !  (C) Copr. 1986-92 Numerical Recipes Software

  real(8) function rj(x,y,z,p)
    real(8), intent(in) :: p,x,y,z
    real(8), parameter :: &
         & ERRTOL=.05d0, C1=3.d0/14.d0,&
         & C2=1.d0/3.d0, C3=3.d0/22.d0, C4=3.d0/26.d0, C5=.75d0*C3, &
         & C6=1.5d0*C4, C7=.5d0*C2, C8=C3+C3

    real(8) :: a,alamb,alpha,ave,b,beta,delp,delx,dely,delz,ea,eb,ec,ed,ee,&
         & fac,pt,rcx,rho,sqrtx,sqrty,sqrtz,sum,tau,xt,yt,zt

    sum=0.d0
    fac=1.d0
    if(p > 0.d0) then
       xt=x
       yt=y
       zt=z
       pt=p
    else
       xt=min(x,y,z)
       zt=max(x,y,z)
       yt=x+y+z-xt-zt
       a=1.d0/(yt-p)
       b=a*(zt-yt)*(yt-xt)
       pt=yt+b
       rho=xt*zt/yt
       tau=p*pt/yt
       rcx=rc(rho,tau)
    end if

    delx = 1.d3
    do while(max(abs(delx),abs(dely),abs(delz),abs(delp)) > ERRTOL)
       sqrtx=sqrt(xt)
       sqrty=sqrt(yt)
       sqrtz=sqrt(zt)
       alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
       alpha=(pt*(sqrtx+sqrty+sqrtz)+sqrtx*sqrty*sqrtz)**2
       beta=pt*(pt+alamb)**2
       sum=sum+fac*rc(alpha,beta)
       fac=.25d0*fac
       xt=.25d0*(xt+alamb)
       yt=.25d0*(yt+alamb)
       zt=.25d0*(zt+alamb)
       pt=.25d0*(pt+alamb)
       ave=.2d0*(xt+yt+zt+pt+pt)
       delx=(ave-xt)/ave
       dely=(ave-yt)/ave
       delz=(ave-zt)/ave
       delp=(ave-pt)/ave
    end do

    ea=delx*(dely+delz)+dely*delz
    eb=delx*dely*delz
    ec=delp**2
    ed=ea-3.d0*ec
    ee=eb+2.d0*delp*(ea-ec)
    rj=3.d0*sum+fac*(1.d0+ed*(-C1+C5*ed-C6*ee)+eb*(C7+delp*(-C8+delp*C4))+delp*ea*(C2-delp*C3)-C2*delp*ec)/(ave*sqrt(ave))
    if (p < 0.d0) rj=a*(b*rj+3.d0*(rcx-rf(xt,yt,zt)))
  end function rj
  !  (C) Copr. 1986-92 Numerical Recipes Software

  real(8) function rf(x,y,z)
    real(8), intent(in) :: x,y,z
    real(8), parameter :: ERRTOL=.08d0,THIRD=1.d0/3.d0,C1=1.d0/24.d0,C2=.1d0,C3=3.d0/44.d0,C4=1.d0/14.d0
    real(8) :: alamb,ave,delx,dely,delz,e2,e3,sqrtx,sqrty,sqrtz,xt,yt,zt
    xt=x
    yt=y
    zt=z

    delx = 1.d3
    do while(max(abs(delx),abs(dely),abs(delz)) > ERRTOL)
       sqrtx=sqrt(xt)
       sqrty=sqrt(yt)
       sqrtz=sqrt(zt)
       alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
       xt=.25d0*(xt+alamb)
       yt=.25d0*(yt+alamb)
       zt=.25d0*(zt+alamb)
       ave=THIRD*(xt+yt+zt)
       delx=(ave-xt)/ave
       dely=(ave-yt)/ave
       delz=(ave-zt)/ave
    end do

    e2=delx*dely-delz**2
    e3=delx*dely*delz
    rf=(1.d0+(C1*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave)
  end function rf


  real(8) function ellec(k)
    implicit none
    real(8), intent(in) :: k
    real(8), parameter :: &
         & a1=0.44325141463d0, &
         & a2=0.06260601220d0, &
         & a3=0.04757383546d0, &
         & a4=0.01736506451d0, &
         & b1=0.24998368310d0, &
         & b2=0.09200180037d0, &
         & b3=0.04069697526d0, &
         & b4=0.00526449639d0
    real(8) :: m1,ee1,ee2

    m1=1.d0-k*k
    ee1=1.d0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*log(1.d0/m1)
    ellec=ee1+ee2
  end function ellec

  real(8) function ellk(k)
    implicit none
    real(8), intent(in) :: k
    real(8), parameter :: &
         & a0=1.38629436112d0, &
         & a1=0.09666344259d0, &
         & a2=0.03590092383d0, &
         & a3=0.03742563713d0, &
         & a4=0.01451196212d0, &
         & b0=0.5d0, &
         & b1=0.12498593597d0, &
         & b2=0.06880248576d0, &
         & b3=0.03328355346d0, &
         & b4=0.00441787012d0
    real(8) :: ek1,ek2,m1

    m1=1.d0-k*k
    ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*log(m1)
    ellk=ek1-ek2
  end function ellk
end module ma_quad
