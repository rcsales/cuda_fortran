module calcPi
contains
    attributes(global) subroutine pi_darts(x, y, results, N)
    use cudafor
    implicit none
        integer :: id
        integer, value :: N
        real, dimension(N) :: x, y, results
        real :: z

        id = (blockIdx%x-1)*blockDim%x + threadIdx%x

        if (id .lt. N) then
            ! SQRT NOT NEEDED, SQRT(1) === 1
            ! Anything above and below 1 would stay the same even with the applied
            ! sqrt function. Therefore using the sqrt function wastes GPU time.
            !z = 1.0
            z = x(id)*x(id)+y(id)*y(id)
            !if (z .lt. 1.0) then
            !   z = 1.0
            !else
            !   z = 0.0
            !endif
            results(id) = z
        endif
    end subroutine pi_darts
end module calcPi

program final_project
    use calcPi
    use cudafor
    implicit none
    integer, parameter :: N = 400
    integer :: i
    real, dimension(N) :: x, y, pi_parts
    real, dimension(N), device :: x_d, y_d, pi_parts_d
    type(dim3) :: grid, tBlock

    ! Initialize the random number generaters seed
    call random_seed()

    ! Make sure we initialize the parts with 0
    pi_parts = 0

    ! Prepare the random numbers (These cannot be generated from inside the
    ! cuda kernel)
    call random_number(x)
    call random_number(y)

    !write(*,*) x, y

    ! Convert the random numbers into graphics card memory land!
    x_d = x
    y_d = y
    pi_parts_d = pi_parts

    ! For the cuda kernel
    tBlock = dim3(256,1,1)
    grid = dim3((N/tBlock%x)+1,1,1)

    ! Start the cuda kernel
    call pi_darts<<<grid, tblock>>>(x_d, y_d, pi_parts_d, N)

    ! Transform the results into CPU Memory
    pi_parts = pi_parts_d
    write(*,*) pi_parts

    write(*,*) 'PI: ', 4.0*sum(pi_parts)/N
end program final_project