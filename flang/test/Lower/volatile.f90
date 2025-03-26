! RUN: bbc %s -o - | FileCheck %s

program p
  integer,volatile::i,arr(10)
  i=0
  arr=1
  ! casting from volatile ref to non-volatile ref should be okay here
  call not_declared_volatile_in_this_scope(i)
  call not_declared_volatile_in_this_scope(arr)
  call declared_volatile_in_this_scope(arr,10)
  print*,arr,i
contains
  elemental subroutine not_declared_volatile_in_this_scope(v)
    integer,intent(inout)::v
    v=1
  end subroutine
  subroutine declared_volatile_in_this_scope(v,n)
    integer,intent(in)::n
    integer,volatile,intent(inout)::v(n)
    v=1
  end subroutine
end program

! CHECK-LABEL: _QQmain
! CHECK:           %[[VAL_10:.*]] = fir.volatile_cast %{{.+}} : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare {{.+}} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[VAL_12:.*]] = fir.alloca i32
! CHECK:           %[[VAL_13:.*]] = fir.volatile_cast %[[VAL_12]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare {{.+}} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           hlfir.assign %{{.+}} to %[[VAL_14]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %{{.+}} to %[[VAL_11]]#0 : i32, !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_15:.*]] = fir.volatile_cast %[[VAL_14]]#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
! CHECK:           fir.call @_QFPnot_declared_volatile_in_this_scope(%[[VAL_15]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           %[[VAL_19:.*]] = hlfir.designate %{{.+}} (%{{.+}})  : (!fir.ref<!fir.array<10xi32>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_20:.*]] = fir.volatile_cast %[[VAL_19]] : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
! CHECK:           fir.call @_QFPnot_declared_volatile_in_this_scope(%[[VAL_20]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           %[[VAL_23:.*]] = fir.volatile_cast %[[VAL_11]]#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[VAL_25:.*]]:3 = hlfir.associate %{{.+}} {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           fir.call @_QFPdeclared_volatile_in_this_scope(%[[VAL_24]], %[[VAL_25]]#0) fastmath<contract> : (!fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_25]]#1, %[[VAL_25]]#2 : !fir.ref<i32>, i1
! CHECK:           %[[VAL_26:.*]] = fir.address_of(@_QQclX6951b66b308fd310127f64e03dcd1051) : !fir.ref<!fir.char<1,78>>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,78>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_28:.*]] = fir.call @_FortranAioBeginExternalListOutput(
! CHECK:           %[[VAL_29:.*]] = fir.embox %{{.+}} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_30:.*]] = fir.volatile_cast %[[VAL_29]] : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
! CHECK:           %[[VAL_32:.*]] = fir.call @_FortranAioOutputDescriptor(
! CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<i32, volatile>
! CHECK:           %[[VAL_34:.*]] = fir.call @_FortranAioOutputInteger32(
! CHECK:           %[[VAL_35:.*]] = fir.call @_FortranAioEndIoStatement(
! CHECK:           return
