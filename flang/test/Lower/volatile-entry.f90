! RUN: bbc %s -o - | FileCheck %s

function f
entry e
volatile f
e=1
end function
function f2
entry e2
volatile e2
e2=1
end function

! CHECK-LABEL:   func.func @_QPf() -> f32 {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = fir.alloca f32 {bindc_name = "f", uniq_name = "_QFfEf"}
! CHECK:           %[[VAL_3:.*]] = fir.volatile_cast %[[VAL_2]] : (!fir.ref<f32>) -> !fir.ref<f32, volatile>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFfEf"} : (!fir.ref<f32, volatile>) -> (!fir.ref<f32, volatile>, !fir.ref<f32, volatile>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 {uniq_name = "_QFfEe"} : (!fir.ref<f32, volatile>) -> (!fir.ref<f32, volatile>, !fir.ref<f32, volatile>)
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_5]]#0 : f32, !fir.ref<f32, volatile>
! CHECK:           %[[VAL_6:.*]] = fir.volatile_cast %[[VAL_4]]#0 : (!fir.ref<f32, volatile>) -> !fir.ref<f32>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<f32>
! CHECK:           return %[[VAL_7]] : f32
! CHECK:         }

! CHECK-LABEL:   func.func @_QPe() -> f32 {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = fir.alloca f32 {bindc_name = "f", uniq_name = "_QFfEf"}
! CHECK:           %[[VAL_3:.*]] = fir.volatile_cast %[[VAL_2]] : (!fir.ref<f32>) -> !fir.ref<f32, volatile>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFfEf"} : (!fir.ref<f32, volatile>) -> (!fir.ref<f32, volatile>, !fir.ref<f32, volatile>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 {uniq_name = "_QFfEe"} : (!fir.ref<f32, volatile>) -> (!fir.ref<f32, volatile>, !fir.ref<f32, volatile>)
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_5]]#0 : f32, !fir.ref<f32, volatile>
! CHECK:           %[[VAL_6:.*]] = fir.volatile_cast %[[VAL_5]]#0 : (!fir.ref<f32, volatile>) -> !fir.ref<f32>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<f32>
! CHECK:           return %[[VAL_7]] : f32
! CHECK:         }

! CHECK-LABEL:   func.func @_QPf2() -> f32 {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = fir.alloca f32 {bindc_name = "f2", uniq_name = "_QFf2Ef2"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFf2Ef2"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_3]]#0 : (!fir.ref<f32>) -> !fir.ref<f32, volatile>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFf2Ee2"} : (!fir.ref<f32, volatile>) -> (!fir.ref<f32, volatile>, !fir.ref<f32, volatile>)
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_5]]#0 : f32, !fir.ref<f32, volatile>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<f32>
! CHECK:           return %[[VAL_6]] : f32
! CHECK:         }

! CHECK-LABEL:   func.func @_QPe2() -> f32 {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = fir.alloca f32 {bindc_name = "f2", uniq_name = "_QFf2Ef2"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFf2Ef2"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_3]]#0 : (!fir.ref<f32>) -> !fir.ref<f32, volatile>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFf2Ee2"} : (!fir.ref<f32, volatile>) -> (!fir.ref<f32, volatile>, !fir.ref<f32, volatile>)
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_5]]#0 : f32, !fir.ref<f32, volatile>
! CHECK:           %[[VAL_6:.*]] = fir.volatile_cast %[[VAL_5]]#0 : (!fir.ref<f32, volatile>) -> !fir.ref<f32>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<f32>
! CHECK:           return %[[VAL_7]] : f32
! CHECK:         }

