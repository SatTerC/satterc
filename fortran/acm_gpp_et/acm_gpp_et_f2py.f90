! f2py-facing shim over ACM-GPP-ET's CARBON_MODEL, for one pixel.
!
! Upstream's own entry point, racmgppet in ACM_GPP_ET_R_interface.f90, is shaped
! for R's .Fortran: every array, outputs included, is passed in pre-allocated,
! and all sizes are explicit arguments. This shim gives f2py a signature it can
! turn into an idiomatic Python function instead:
!
!   - inputs are intent(in), outputs intent(out), so Python passes the drivers
!     and gets the result arrays back, allocated by the wrapper;
!   - the size arguments come last and are inferred from the array shapes;
!   - status reports the one failure mode the caller can trigger (see below),
!     because a Fortran `stop` would take the whole Python process down with it.
!
! CARBON_MODEL keeps state in module variables, which persist for the lifetime
! of the loaded extension, i.e. the Python process. Most are reset on every
! call, but deltat_1, wSWP_time and soilwatermm are allocated once, sized by
! the nodays of the *first* call, and never reallocated. A later call with a
! different nodays would read stale entries or index past their end, so it is
! refused with status = 1.

subroutine acm_pixel(met, pars, lat, soil_frac_clay_in, soil_frac_sand_in, &
                     fluxes, soil_water, wswp, status, nomet, nodays, nopars)

  use CARBON_MODEL_MOD, only: CARBON_MODEL, wSWP_time, nos_soil_layers, &
                              soil_frac_clay, soil_frac_sand

  implicit none

  integer, parameter :: nofluxes = 8, nopools = 1

  integer, intent(in) :: nomet, nodays, nopars
  double precision, intent(in) :: met(nomet, nodays)  ! drivers, see CARBON_MODEL
  double precision, intent(in) :: pars(nopars)        ! parameters, ditto
  double precision, intent(in) :: lat                 ! latitude (degrees)
  ! f2py cannot see nos_soil_layers when it scans this file, so the layer count
  ! is written as a literal; the check below keeps the two in step.
  double precision, intent(in) :: soil_frac_clay_in(4), soil_frac_sand_in(4)

  double precision, intent(out) :: fluxes(nodays, nofluxes)
  double precision, intent(out) :: soil_water(nodays)  ! top soil layer (mm)
  double precision, intent(out) :: wswp(nodays)        ! weighted SWP (MPa)
  integer, intent(out) :: status

  double precision :: deltat(nodays), pools(nodays + 1, nopools)
  integer :: n

  status = 0
  if (nos_soil_layers /= 4) then
    status = 2
    return
  end if
  if (allocated(wSWP_time)) then
    if (size(wSWP_time) /= nodays) then
      status = 1
      return
    end if
  end if

  soil_frac_clay = soil_frac_clay_in
  soil_frac_sand = soil_frac_sand_in

  ! As racmgppet: the time step is the difference of the run-day driver.
  deltat(1) = met(1, 1)
  do n = 2, nodays
    deltat(n) = met(1, n) - met(1, n - 1)
  end do

  fluxes = 0d0
  pools = 0d0
  call CARBON_MODEL(1, nodays, met, pars, deltat, nodays, lat, fluxes, pools, &
                    nopars, nomet, nopools, nofluxes)

  soil_water = pools(1:nodays, 1)
  wswp = wSWP_time(1:nodays)

end subroutine acm_pixel
