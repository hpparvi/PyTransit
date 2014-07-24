module ma_quad
  use omp_lib
  implicit none

  real(8), parameter :: PI = acos(-1.d0)
  real(8), parameter :: HALF_PI = 0.5d0*PI
  real(8), parameter :: INV_PI = 1.d0/PI

contains
  subroutine eval(z0,k,u,c,nthr,nz,flux)
    implicit none
    integer, intent(in) :: nthr, nz
    real(8), intent(in), dimension(nz) :: z0
    real(8), intent(in) :: u(2), c
    real(8), intent(out), dimension(nz) :: flux
    real(8), dimension(nz) :: mu, lambdad, etad, lambdae
    real(8) :: k,k2,z2,lam,x1,x2,x3,z,omega,kap0,kap1,q,Kk,Ek,Pk,n
    integer :: i

    real(8), dimension(nz) :: ztmp, ftmp
    logical, dimension(nz) :: mask
    integer :: npt_t

    if(abs(k-0.5) < 1.d-3) then 
       k=0.5
    end if

    k2 = k**2
    omega=1.-u(1)/3.d0-u(2)/6.d0

    mask = (z0 > 0.d0) .and. (z0 < 1.d0+k)
    npt_t = count(mask)    
    ztmp(1:npt_t) = pack(z, mask)

    !$ call omp_set_num_threads(nthr)
    !$omp parallel do default(none) shared(z0,flux,u,c,omega,k,k2,nz,npt_t,ztmp,ftmp,mu, lambdad, etad, lambdae) &
    !$omp private(z2,lam,x1,x2,x3,z,kap0,kap1,q,Kk,Ek,Pk,n)
    do i=1,npt_t
       z=ztmp(i)
       z2 = z**2
       x1=(k-z)**2
       x2=(k+z)**2
       x3=k**2-z**2

       ! the source is unocculted:
       ! Table 3, I.
       if(z > 1.0+k) then
          lambdad(i)=0.0
          etad(i)=0.0
          lambdae(i)=0.0

       ! the  source is completely occulted:
       ! Table 3, II.
       else if(k > 1.0 .and. z < k-1.0) then
          lambdad(i)=1.0
          etad(i)=1.0
          lambdae(i)=1.0

       ! the source is partly occulted and the occulting object crosses the limb:
       ! Equation (26):
       else if(z > abs(1.0-k) .and. z < 1.0+k) then
          kap1=acos(min((1.0-k2+z2)/2.0/z,1.0))
          kap0=acos(min((k2+z2-1.0)/2.0/k/z,1.0))
          lambdae(i) = k2*kap0+kap1
          lambdae(i) = (lambdae(i)-0.50*sqrt(max(4.0*z2-(1.0+z2-k2)**2,0.0)))*INV_PI
       end if

       ! the occulting object transits the source star (but doesn't completely cover it):
       if(z < 1.0-k) then
          lambdae(i)=k2
       end if

       ! the edge of the occulting star lies at the origin- special expressions in this case:
       if(abs(z-k) < 1.d-4*(z+k)) then
          ! Table 3, Case V.:
          if(z > 0.50) then
             lam=HALF_PI
             q=0.50/k
             Kk=ellk(q)
             Ek=ellec(q)
             lambdad(i)=1.0/3.0+16.0*k/9.0*INV_PI*(2.0*k2-1.0)*Ek-(32.0*k**4-20.0*k2+3.0)/9.0*INV_PI/k*Kk
             etad(i)=1.0/2.0*INV_PI*(kap1+k2*(k2+2.0*z2)*kap0-(1.0+5.0*k2+z2)/4.0*sqrt((1.0-x1)*(x2-1.0)))
             if(k.eq.0.50) then
                lambdad(i)=1.0/3.0-4.0*INV_PI/9.0
                etad(i)=3.0/32.0
             end if
          else
             ! Table 3, Case VI.:
             lam=HALF_PI
             q=2.0*k
             Kk=ellk(q)
             Ek=ellec(q)
             lambdad(i)=1.0/3.0+2.0/9.0*INV_PI*(4.0*(2.0*k2-1.0)*Ek+(1.0-4.0*k2)*Kk)
             etad(i)=k2/2.0*(k2+2.0*z2)
          end if
       end if

       ! the occulting star partly occults the source and crosses the limb:
       ! Table 3, Case III:
       if((z > 0.50+abs(k-0.50) .and. z < 1.0+k).or.(k > 0.50 .and. z > abs(1.0-k)*1.00010 .and. z < k)) then
          lam=HALF_PI
          q=sqrt((1.0-(k-z)**2)/4.0/z/k)
          Kk=ellk(q)
          Ek=ellec(q)
          n=1.0/x1-1.0
          Pk=Kk-n/3.0*rj(0.d0, 1.0-q*q, 1.0d0, 1.0+n)
          lambdad(i)=1.0/9.0*INV_PI/sqrt(k*z)*(((1.0-x2)*(2.0*x2 &
          &        +x1-3.0)-3.0*x3*(x2-2.0))*Kk+4.0*k*z*(z2 &
          &        +7.0*k2-4.0)*Ek-3.0*x3/x1*Pk)
          if(z < k) then
             lambdad(i)=lambdad(i)+2.0/3.0
          end if
          etad(i)=1.0/2.0*INV_PI*(kap1+k2*(k2+2.0*z2)*kap0-(1.0+5.0*k2+z2)/4.0*sqrt((1.0-x1)*(x2-1.0)))
       end if

       ! the occulting star transits the source:
       ! Table 3, Case IV.:
       if(k < 1.0 .and. z < (1.0-k)*1.00010) then
          lam=HALF_PI
          q=sqrt((x2-x1)/(1.0-x1))
          Kk=ellk(q)
          Ek=ellec(q)
          n=x2/x1-1.0
          Pk=Kk-n/3.0*rj(0.0d0,1.0-q*q,1.0d0,1.0+n)
          lambdad(i)=2.0/9.0*INV_PI/sqrt(1.0-x1)*((1.0-5.0*z2+k2+&
          &         x3*x3)*Kk+(1.0-x1)*(z2+7.0*k2-4.0)*Ek-3.0*x3/x1*Pk)
          if(z < k) lambdad(i)=lambdad(i)+2.0/3.0
          if(abs(k+z-1.0) < 1.d-4) then
             lambdad(i)=2.0/3.0*INV_PI*acos(1.0-2.0*k)-4.0/9.0*INV_PI*&
             &            sqrt(k*(1.0-k))*(3.0+2.0*k-8.0*k2)
          end if
          etad(i)=k2/2.0*(k2+2.0*z2)
       end if
       ftmp(i) = 1.0-((1.0-u(1)-2.0*u(2))*lambdae(i)+(u(1)+2.0*u(2))*lambdad(i)+u(2)*etad(i))/omega
       ftmp(i) = c + (1.0-c)*flux(i)
    end do
    !$omp end parallel do
    
  end subroutine eval

  real(8) function rc(x,y)
    real(8), intent(in) :: x,y
    real(8), parameter :: ERRTOL=0.040, THIRD=1.0/3.0, C1=0.30, C2=1.0/7.0, C3=0.3750, C4=9.0/22.0
    real(8) :: alamb,ave,s,w,xt,yt

    if(y > 0.0) then
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
       alamb=2.0*sqrt(xt)*sqrt(yt)+yt
       xt=0.250*(xt+alamb)
       yt=0.250*(yt+alamb)
       ave=THIRD*(xt+yt+yt)
       s=(yt-ave)/ave
    end do

    rc=w*(1.0+s*s*(C1+s*(C2+s*(C3+s*C4))))/sqrt(ave)
  end function rc

  real(8) function rj(x,y,z,p)
    real(8), intent(in) :: p,x,y,z
    real(8), parameter :: &
         & ERRTOL=.050, C1=3.0/14.0,&
         & C2=1.0/3.0, C3=3.0/22.0, C4=3.0/26.0, C5=.750*C3, &
         & C6=1.50*C4, C7=.50*C2, C8=C3+C3

    real(8) :: a,alamb,alpha,ave,b,beta,delp,delx,dely,delz,ea,eb,ec,ed,ee,&
         & fac,pt,rcx,rho,sqrtx,sqrty,sqrtz,sum,tau,xt,yt,zt

    sum=0.0
    fac=1.0
    if(p > 0.0) then
       xt=x
       yt=y
       zt=z
       pt=p
    else
       xt=min(x,y,z)
       zt=max(x,y,z)
       yt=x+y+z-xt-zt
       a=1.0/(yt-p)
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
       fac=.250*fac
       xt=.250*(xt+alamb)
       yt=.250*(yt+alamb)
       zt=.250*(zt+alamb)
       pt=.250*(pt+alamb)
       ave=.20*(xt+yt+zt+pt+pt)
       delx=(ave-xt)/ave
       dely=(ave-yt)/ave
       delz=(ave-zt)/ave
       delp=(ave-pt)/ave
    end do

    ea=delx*(dely+delz)+dely*delz
    eb=delx*dely*delz
    ec=delp**2
    ed=ea-3.0*ec
    ee=eb+2.0*delp*(ea-ec)
    rj=3.0*sum+fac*(1.0+ed*(-C1+C5*ed-C6*ee)+eb*(C7+delp*(-C8+delp*C4))+delp*ea*(C2-delp*C3)-C2*delp*ec)/(ave*sqrt(ave))
    if (p < 0.0) rj=a*(b*rj+3.0*(rcx-rf(xt,yt,zt)))
  end function rj

  real(8) function rf(x,y,z)
    real(8), intent(in) :: x,y,z
    real(8), parameter :: ERRTOL=.080,THIRD=1.0/3.0,C1=1.0/24.0,C2=.10,C3=3.0/44.0,C4=1.0/14.0
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
       xt=.250*(xt+alamb)
       yt=.250*(yt+alamb)
       zt=.250*(zt+alamb)
       ave=THIRD*(xt+yt+zt)
       delx=(ave-xt)/ave
       dely=(ave-yt)/ave
       delz=(ave-zt)/ave
    end do

    e2=delx*dely-delz**2
    e3=delx*dely*delz
    rf=(1.0+(C1*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave)
  end function rf


  real(8) function ellec(k)
    implicit none
    real(8), intent(in) :: k
    real(8), parameter :: &
         & a1=0.443251414630, &
         & a2=0.062606012200, &
         & a3=0.047573835460, &
         & a4=0.017365064510, &
         & b1=0.249983683100, &
         & b2=0.092001800370, &
         & b3=0.040696975260, &
         & b4=0.005264496390
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
         & a0=1.386294361120, &
         & a1=0.096663442590, &
         & a2=0.035900923830, &
         & a3=0.037425637130, &
         & a4=0.014511962120, &
         & b0=0.50, &
         & b1=0.124985935970, &
         & b2=0.068802485760, &
         & b3=0.033283553460, &
         & b4=0.004417870120
    real(8) :: ek1,ek2,m1

    m1=1.0-k*k
    ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*log(m1)
    ellk=ek1-ek2
  end function ellk
end module ma_quad
