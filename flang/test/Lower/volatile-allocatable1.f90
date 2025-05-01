! Test array allocatable volatiles
subroutine test_array_volatile()
  type :: base_type
    integer :: i = 42
  end type
  type, extends(base_type) :: ext_type
    integer :: j = 100
  end type
!   class(base_type), allocatable, volatile :: v1(:)
!   type(comp_type), allocatable, volatile :: v3(:)
!   character(len=:), allocatable, volatile :: c1(:)
!   integer, allocatable, volatile :: i1(:)
  type(ext_type), allocatable, volatile :: v2(:,:)
  
!   allocate(v1(3))
!   do i = 1, 3
!     if (v1(i)%i /= 42) print *, 'error: v1(i)%i not initialized'
!   end do
  
!   ! 2D allocation with polymorphic type
!   allocate(ext_type :: v1(2))
!   select type (v1)
!     type is (ext_type)
!       if (v1(1)%j /= 100) print *, 'error: v1(1)%j not initialized'
!   end select
  
  ! 2D array allocation
  allocate(v2(2,3))
  if (v2(1,1)%i /= 42) print *, 'error: v2(1,1)%i not initialized'
  
!   ! Array of derived type with components
!   allocate(v3(2))
!   if (v3(1)%str /= "test") print *, 'error: v3(1)%str not initialized'
!   if (any(v3(2)%arr /= [1, 2])) print *, 'error: v3(2)%arr not initialized'
  
!   allocate(character(15) :: c1(3))
!   c1(1) = "array1"
!   c1(2) = "array2"
!   c1(3) = "array3"
!   if (c1(2) /= "array2") print *, 'error: c1(2) value incorrect'
  
!   ! Integer array with source
!   allocate(i1(5), source=[10, 20, 30, 40, 50])
!   if (any(i1 /= [10, 20, 30, 40, 50])) print *, 'error: i1 not initialized correctly'
  
!   deallocate(v1, v2, v3, c1, i1)
end subroutine
