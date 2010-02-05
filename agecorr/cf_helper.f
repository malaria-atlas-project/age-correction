! Author: Anand Patil
! Date: 6 Feb 2009
! License: Creative Commons BY-NC-SA
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      SUBROUTINE cfh(k,q,s,nk,nq,o)
cf2py intent(hide) nk,nq
cf2py intent(out) o
      LOGICAL s(nq)
      INTEGER nk, nq, i, j 
      DOUBLE PRECISION k(nk), q(nq), o(nk)
      
      do i=1,nk
          o(i) = 0.0D0
          do j=1,nq
              if (s(j)) then
                  o(i) = o(i) + dlog(1.0D0-k(i)*q(j))
              end if
          end do
      end do
      
      RETURN
      END
      

      SUBROUTINE cfhs(k,q,s,nq,o)
cf2py intent(hide) nq
cf2py intent(out) o
      LOGICAL s(nq)
      INTEGER nq, j 
      DOUBLE PRECISION k, q(nq), o
      
      o = 0.0D0
      do j=1,nq
          if (s(j)) then
              o = o + dlog(1.0D0-k*q(j))
          end if
      end do
      
      RETURN
      END
      