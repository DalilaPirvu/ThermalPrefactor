module constants
  integer, parameter :: dl = kind(1.d0)
  real(dl), parameter :: twopi =  6.2831853071795864769252867665590

  !> Initialise fluctuations in linear perturbation theory approximation
  !> type : (1) sets vacuum
  !>        (2) thermal+vacuum
  !>        (3) only thermal
  !>        (4) rayleigh jeans
  integer, parameter :: type = 4
  integer, parameter :: seedfac = 8

  integer, parameter :: nLat     = 1024
  integer, parameter :: nTimeMax = 3*nLat
  integer, parameter :: lSim = 1350, nSim = 1360

  real(dl), parameter :: gcpl  = 0.3_dl !0.1_dl
  real(dl), parameter :: m2    = 1._dl
  real(dl), parameter :: temp  = 10._dl !50._dl

  real(dl), parameter :: m2eff = m2

  real(dl), parameter :: gsqinv = 1._dl/gcpl**2.
  real(dl), parameter :: gsq = gcpl**2.

  real(dl), parameter :: lenLat = 10._dl
  real(dl), parameter :: dx   = lenLat/dble(nLat)
  real(dl), parameter :: dk   = twopi/lenLat
  real(dl), parameter :: alph = 16._dl

  real(dl), parameter :: fldinit = 0._dl
  real(dl), parameter :: mominit = 0._dl

  integer, parameter :: kspec = nLat/2
  integer, parameter :: nFld  = 1
  integer, parameter :: nVar  = 2*nFld*nLat+1

  ! Normalise assuming the Box-Mueller transform gives a complex random deviate with unit variance
  real(dl), parameter :: norm = 1._dl / sqrt(2._dl * lenLat)  

!!!!!!!!!!!!!!!!!!!!!!
! define box average <cos(phi)> = c = -0.7 as the moment the decay took place
! from that moment, we evolve another nLat/2 timesteps so that tv fills whole volume
! interrupt evolution if no decay occured up to nTimeMax
!!!!!!!!!!!!!!!!!!!!!!

end module constants
