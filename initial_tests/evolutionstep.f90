  subroutine time_evolve(sim, temp, alp, m2eff) 
    real(dl) :: m2eff
    real(dl) :: temp
    real(dl) :: dt
    real(dl) :: dtout
    real(dl) :: treshold
    real(dl) :: alp, fld_amp, fld_amp_max
    integer :: j, k, m, ff
    integer :: sim
    integer :: msave
    integer :: width
    integer :: tclock
    logical :: bool1, bool2
    real(dl), dimension(1:size(fld(:,1)), 2) :: df

    df = fld(:,:)

    dt = dx / alp
!    if (dt > dx) print*, "Warning, violating Courant condition"
    dtout = dt * alp
    tclock = 0

    k = 1
    bool1 = .True.
    bool2 = .False.
    treshold = 1.5
    width = 30

    do while ( bool1 )
       do j = 1, int(alp)
          call gl10(yvec, dt)
       end do

       if ( m2 > m2eff .AND. .NOT. bool2 ) then 
          fld_amp_max = 0.
          msave = 0
          do m = width+1, nLat-width, width/3
             fld_amp = 0.
             do ff = -width, +width
                fld_amp = fld_amp + fld(m+ff,1)
             enddo
             fld_amp = fld_amp/(2*ff+1)
             if ( abs(fld_amp) > abs(fld_amp_max) ) then
                fld_amp_max = fld_amp
                msave = m
             endif
          enddo
!          print*, "fld_amp_max, x, t = ", fld_amp_max, msave, k

          if ( abs(fld_amp_max) > treshold ) then
             bool2 = .True.
             tclock = k
             call output_fields(df, dtout, sim, temp, m2eff, tclock)
             print*, "fld_amp_max, x, tstart = ", fld_amp_max, msave, tclock
          endif
       else
!          print*, "output time = ", k
          do m = 1, nLat
             if ( abs(fld(m,1)) > 10. ) then
                bool1 = .False.
                exit
             endif
          enddo
          call output_fields(fld, dtout, sim, temp, m2eff, tclock)
       endif

       k = k + 1
       if ( k == nTimeMax) then
          if ( .NOT. bool2 ) then
             tclock = k
             call output_fields(df, dtout, sim, temp, m2eff, tclock)
             call output_fields(fld, dtout, sim, temp, m2eff, tclock)
             print*, "undecayed"
          end if
          bool1 = .False.
          exit
       end if
    end do

 !   do while ( bool1 )
 !      do j = 1, int(alp)
 !         call gl10(yvec, dt)
 !      end do
 !      k = k + 1
 !      if ( k == nTimeMax) then
 !         tclock = k
 !         call output_fields(df, dtout, sim, temp, m2eff, tclock)
 !         call output_fields(fld, dtout, sim, temp, m2eff, tclock)
 !         print*, "undecayed"
 !         exit
 !      end if
 !   end do
  end subroutine time_evolve

