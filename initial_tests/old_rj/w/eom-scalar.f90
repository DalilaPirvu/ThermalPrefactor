#include "macros.h"
#include "fldind.h"
#define FIELD_TYPE Field_Model

module eom

  use constants
  use fftw3
  implicit none

  real(dl), dimension(1:nVar), target :: yvec
  type(transformPair1D) :: tPair

contains

  subroutine derivs(yc,yp)
    real(dl), dimension(:), intent(in)  :: yc
    real(dl), dimension(:), intent(out) :: yp

    yp(TIND) = 1._dl

! free:
!    yp(DFLD) = - m2eff*yc(FLD)
! driven:
!    yp(DFLD) = - yc(FLD) - yc(FLD)**3.
! false vacuum, unbounded:
    yp(DFLD) = - yc(FLD) + yc(FLD)**3.

    tPair%realSpace(:) = yc(FLD)
    call laplacian_1d_wtype(tPair,dk)
    yp(DFLD) = yp(DFLD) + tPair%realSpace(:)

    yp(FLD)  = yc(DFLD)

  end subroutine derivs

end module eom
