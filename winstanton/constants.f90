module constants
  integer, parameter :: dl = kind(1.d0)
  real(dl), parameter :: twopi =  6.2831853071795864769252867665590

  integer, parameter :: nLat     = 2048
  integer, parameter :: nTimeMax = 2048
  integer, parameter :: lSim = 0, nSim = 1

  real(dl), parameter :: m2    = 1.
  integer,  parameter :: seedfac = 1
  real(dl), parameter :: temp  = 0.13
  real(dl), parameter :: lenLat= 82

! if free field
!  real(dl), parameter :: m2eff = m2
! if positive curvature
!  real(dl), parameter :: m2eff = m2 + temp*3./2.
! if negative curvature
  real(dl), parameter :: m2eff = m2 - temp*3./2.

  real(dl), parameter :: dx  = lenLat/nLat
  real(dl), parameter :: dk  = twopi/lenLat
  real(dl), parameter :: alph = 6.   ! courrant number

  real(dl), parameter :: fldinit = 0.
  real(dl), parameter :: mominit = 0.

  integer, parameter :: kspec = nLat/2
  integer, parameter :: nFld  = 1
  integer, parameter :: nVar  = 2*nFld*nLat+1

  real(dl), parameter :: norm = 1. / sqrt(nLat * dx)

end module constants
