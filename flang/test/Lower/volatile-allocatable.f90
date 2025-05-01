! RUN: bbc %s -o - | FileCheck %s

module derived_types
  type :: base_type
    integer :: i = 42
  end type

  type, extends(base_type) :: ext_type
    integer :: j = 100
  end type

  type :: comp_type
    character(10) :: str = "test"
    integer :: arr(2) = [1, 2]
  end type
end module

subroutine test_scalar_volatile()
  use derived_types
  class(base_type), allocatable, volatile :: v1
  type(ext_type), allocatable, volatile :: v2
  type(comp_type), allocatable, volatile :: v3
  character(len=:), allocatable, volatile :: c1

  ! Allocation without source
  allocate(v1)
  if (v1%i /= 42) print *, 'error: v1%i not initialized correctly'

  ! Allocate polymorphic derived type with dynamic type
  allocate(ext_type :: v1)
  select type (v1)
    type is (ext_type)
      if (v1%j /= 100) print *, 'error: v1%j not initialized correctly'
  end select

  ! Allocation with source
  allocate(v2, source=ext_type())
  if (v2%i /= 42 .or. v2%j /= 100) print *, 'error: v2 source allocation incorrect'

  ! Deferred-length characters
  allocate(character(20) :: c1)
  c1 = "volatile character"
  if (len(c1) /= 20) print *, 'error: c1 length incorrect'
  if (c1 /= "volatile character") print *, 'error: c1 value incorrect'
  
  ! Allocation with components
  allocate(v3)
  if (v3%str /= "test") print *, 'error: v3%str not initialized'
  if (any(v3%arr /= [1, 2])) print *, 'error: v3%arr not initialized'

  deallocate(v1, v2, v3, c1)
end subroutine

! Test with both volatile and asynchronous attributes
subroutine test_volatile_asynchronous()
  use derived_types
  class(base_type), allocatable, volatile, asynchronous :: v1(:)
  integer, allocatable, volatile, asynchronous :: i1(:)
  
  allocate(v1(4))
  allocate(i1(4), source=[1, 2, 3, 4])
  
  if (v1(1)%i /= 42) print *, 'error: v1(1)%i not initialized'
  if (i1(3) /= 3) print *, 'error: i1(3) not initialized correctly'
  
  deallocate(v1, i1)
end subroutine

! Test with OpenMP
subroutine test_omp_volatile()
  use derived_types
  class(base_type), allocatable, volatile :: v(:)
  
  allocate(v(4))
  
  !$omp parallel private(v)
    allocate(v(2))
    if (v(1)%i /= 42) print *, 'error: thread v(1)%i not initialized'
    
    select type(v)
    class is (base_type)
      v(1)%i = 100
      if (v(1)%i /= 100) print *, 'error: thread v(1)%i not updated'
    end select
    
    deallocate(v)
  !$omp end parallel
  
  if (v(1)%i /= 42) print *, 'error: main v(1)%i not preserved'
  
  deallocate(v)
end subroutine

! Test allocate with mold
subroutine test_mold_allocation()
  use derived_types
  type(comp_type) :: template
  type(comp_type), allocatable, volatile :: v(:)
  
  template%str = "mold test"
  template%arr = [5, 6]
  
  allocate(v(3), mold=template)
  
  if (v(1)%str /= "mold test") print *, 'error: v(1)%str mold failed'
  if (any(v(2)%arr /= [5, 6])) print *, 'error: v(2)%arr mold failed'
  
  deallocate(v)
end subroutine

! Test unlimited polymorphic allocation
subroutine test_unlimited_polymorphic()
  use derived_types
  class(*), allocatable, volatile :: up
  class(*), allocatable, volatile :: upa(:)
  
  ! Scalar allocation
  allocate(integer :: up)
  select type(up)
    type is (integer)
      up = 123
  end select
  
  ! Array allocation with source
  allocate(character(10) :: up)
  select type(up)
    type is (character(*))
      up = "class(*)"
  end select
  
  ! Array allocation
  allocate(real :: upa(3))
  select type(upa)
    type is (real)
      upa = [1.1, 2.2, 3.3]
      if (abs(upa(2) - 2.2) > 1.0e-5) print *, 'error: upa(2) incorrect'
  end select
  
  deallocate(up, upa)
end subroutine

program main
  call test_scalar_volatile()
  call test_array_volatile()
  call test_volatile_asynchronous()
  call test_omp_volatile()
  call test_mold_allocation()
  call test_unlimited_polymorphic()
  print *, 'PASS'
end program
