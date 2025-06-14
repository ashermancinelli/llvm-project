! RUN: bbc -hlfir=false -fwrapv %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPss1()
subroutine ss1
  ! CHECK: %[[aa:[0-9]+]] = fir.alloca !fir.array<2650000xf32> {bindc_name = "aa", uniq_name = "_QFss1Eaa"}
  ! CHECK: %[[shape:[0-9]+]] = fir.shape {{.*}} : (index) -> !fir.shape<1>
  integer, parameter :: N = 2650000
  real aa(N)
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}} : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  aa = -2
  ! CHECK: %[[temp:[0-9]+]] = fir.allocmem !fir.array<2650000xf32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}} : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) {{.*}} : (!fir.heap<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) [{{.*}}] {{.*}} : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) [{{.*}}] {{.*}} : (!fir.heap<!fir.array<2650000xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) {{.*}} : (!fir.heap<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}} : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<2650000xf32>>
  aa(2:N) = aa(1:N-1) + 7.0
! print*, aa(1:2), aa(N-1:N)
end

! CHECK-LABEL:   func.func @_QPss2(
! CHECK-SAME:                      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_0:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_3:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_4:.*]] = arith.constant 7.000000e+00 : f32
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_7:.*]] = arith.constant -2.000000e+00 : f32
! CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_9:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_8]] : index
! CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_8]] : index
! CHECK:           %[[VAL_13:.*]] = fir.alloca !fir.array<?xf32>, %[[VAL_12]] {bindc_name = "aa", uniq_name = "_QFss2Eaa"}
! CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:           cf.br ^bb1(%[[VAL_8]], %[[VAL_12]] : index, index)
! CHECK:         ^bb1(%[[VAL_15:.*]]: index, %[[VAL_16:.*]]: index):
! CHECK:           %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_8]] : index
! CHECK:           cf.cond_br %[[VAL_17]], ^bb2, ^bb3
! CHECK:         ^bb2:
! CHECK:           %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[VAL_6]] : index
! CHECK:           %[[VAL_19:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_14]]) %[[VAL_18]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_19]] : !fir.ref<f32>
! CHECK:           %[[VAL_20:.*]] = arith.subi %[[VAL_16]], %[[VAL_6]] : index
! CHECK:           cf.br ^bb1(%[[VAL_18]], %[[VAL_20]] : index, index)
! CHECK:         ^bb3:
! CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_10]], %[[VAL_0]] : index
! CHECK:           %[[VAL_22:.*]] = arith.cmpi sgt, %[[VAL_21]], %[[VAL_8]] : index
! CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_21]], %[[VAL_8]] : index
! CHECK:           %[[VAL_24:.*]] = fir.slice %[[VAL_1]], %[[VAL_10]], %[[VAL_6]] : (index, index, index) -> !fir.slice<1>
! CHECK:           %[[VAL_25:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_12]]
! CHECK:           cf.br ^bb4(%[[VAL_8]], %[[VAL_12]] : index, index)
! CHECK:         ^bb4(%[[VAL_26:.*]]: index, %[[VAL_27:.*]]: index):
! CHECK:           %[[VAL_28:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_8]] : index
! CHECK:           cf.cond_br %[[VAL_28]], ^bb5, ^bb6
! CHECK:         ^bb5:
! CHECK:           %[[VAL_29:.*]] = arith.addi %[[VAL_26]], %[[VAL_6]] : index
! CHECK:           %[[VAL_30:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_14]]) %[[VAL_29]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_31:.*]] = fir.array_coor %[[VAL_25]](%[[VAL_14]]) %[[VAL_29]] : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_32:.*]] = fir.load %[[VAL_30]] : !fir.ref<f32>
! CHECK:           fir.store %[[VAL_32]] to %[[VAL_31]] : !fir.ref<f32>
! CHECK:           %[[VAL_33:.*]] = arith.subi %[[VAL_27]], %[[VAL_6]] : index
! CHECK:           cf.br ^bb4(%[[VAL_29]], %[[VAL_33]] : index, index)
! CHECK:         ^bb6:
! CHECK:           %[[VAL_34:.*]] = arith.subi %[[VAL_9]], %[[VAL_5]] : i32
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> index
! CHECK:           %[[VAL_36:.*]] = fir.slice %[[VAL_6]], %[[VAL_35]], %[[VAL_6]] : (index, index, index) -> !fir.slice<1>
! CHECK:           cf.br ^bb7(%[[VAL_8]], %[[VAL_23]] : index, index)
! CHECK:         ^bb7(%[[VAL_37:.*]]: index, %[[VAL_38:.*]]: index):
! CHECK:           %[[VAL_39:.*]] = arith.cmpi sgt, %[[VAL_38]], %[[VAL_8]] : index
! CHECK:           cf.cond_br %[[VAL_39]], ^bb8, ^bb9(%[[VAL_8]], %[[VAL_12]] : index, index)
! CHECK:         ^bb8:
! CHECK:           %[[VAL_40:.*]] = arith.addi %[[VAL_37]], %[[VAL_6]] : index
! CHECK:           %[[VAL_41:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_14]]) {{\[}}%[[VAL_36]]] %[[VAL_40]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_42:.*]] = fir.load %[[VAL_41]] : !fir.ref<f32>
! CHECK:           %[[VAL_43:.*]] = arith.addf %[[VAL_42]], %[[VAL_4]] fastmath<contract> : f32
! CHECK:           %[[VAL_44:.*]] = fir.array_coor %[[VAL_25]](%[[VAL_14]]) {{\[}}%[[VAL_24]]] %[[VAL_40]] : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[VAL_43]] to %[[VAL_44]] : !fir.ref<f32>
! CHECK:           %[[VAL_45:.*]] = arith.subi %[[VAL_38]], %[[VAL_6]] : index
! CHECK:           cf.br ^bb7(%[[VAL_40]], %[[VAL_45]] : index, index)
! CHECK:         ^bb9(%[[VAL_46:.*]]: index, %[[VAL_47:.*]]: index):
! CHECK:           %[[VAL_48:.*]] = arith.cmpi sgt, %[[VAL_47]], %[[VAL_8]] : index
! CHECK:           cf.cond_br %[[VAL_48]], ^bb10, ^bb11
! CHECK:         ^bb10:
! CHECK:           %[[VAL_49:.*]] = arith.addi %[[VAL_46]], %[[VAL_6]] : index
! CHECK:           %[[VAL_50:.*]] = fir.array_coor %[[VAL_25]](%[[VAL_14]]) %[[VAL_49]] : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_51:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_14]]) %[[VAL_49]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_52:.*]] = fir.load %[[VAL_50]] : !fir.ref<f32>
! CHECK:           fir.store %[[VAL_52]] to %[[VAL_51]] : !fir.ref<f32>
! CHECK:           %[[VAL_53:.*]] = arith.subi %[[VAL_47]], %[[VAL_6]] : index
! CHECK:           cf.br ^bb9(%[[VAL_49]], %[[VAL_53]] : index, index)
! CHECK:         ^bb11:
! CHECK:           fir.freemem %[[VAL_25]] : !fir.heap<!fir.array<?xf32>>
! CHECK:           %[[VAL_54:.*]] = fir.address_of(@_QQclX88e38683fcd4166fc29b4ce91ec570e4) : !fir.ref<!fir.char<1,80>>
! CHECK:           %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (!fir.ref<!fir.char<1,80>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_56:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_3]], %[[VAL_55]], %[[VAL_2]]) fastmath<contract> {fir.llvm_memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, llvm.nocallback, llvm.nosync} : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_57:.*]] = fir.slice %[[VAL_6]], %[[VAL_1]], %[[VAL_6]] : (index, index, index) -> !fir.slice<1>
! CHECK:           %[[VAL_58:.*]] = fir.embox %[[VAL_13]](%[[VAL_14]]) {{\[}}%[[VAL_57]]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<2xf32>>
! CHECK:           %[[VAL_59:.*]] = fir.convert %[[VAL_58]] : (!fir.box<!fir.array<2xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_60:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_56]], %[[VAL_59]]) fastmath<contract> {llvm.nocallback, llvm.nosync} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_61:.*]] = fir.slice %[[VAL_35]], %[[VAL_10]], %[[VAL_6]] : (index, index, index) -> !fir.slice<1>
! CHECK:           %[[VAL_62:.*]] = fir.embox %[[VAL_13]](%[[VAL_14]]) {{\[}}%[[VAL_61]]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_63:.*]] = fir.convert %[[VAL_62]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_64:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_56]], %[[VAL_63]]) fastmath<contract> {llvm.nocallback, llvm.nosync} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_65:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_56]]) fastmath<contract> {fir.llvm_memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, llvm.nocallback, llvm.nosync} : (!fir.ref<i8>) -> i32
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPss3(
! CHECK-SAME:                      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_0:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_2:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_3:.*]] = arith.constant 7.000000e+00 : f32
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_6:.*]] = arith.constant -2.000000e+00 : f32
! CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_8:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_9:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_7]] : index
! CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_7]] : index
! CHECK:           %[[VAL_13:.*]] = fir.alloca !fir.array<2x?xf32>, %[[VAL_12]] {bindc_name = "aa", uniq_name = "_QFss3Eaa"}
! CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_8]], %[[VAL_12]] : (index, index) -> !fir.shape<2>
! CHECK:           cf.br ^bb1(%[[VAL_7]], %[[VAL_12]] : index, index)
! CHECK:         ^bb1(%[[VAL_15:.*]]: index, %[[VAL_16:.*]]: index):
! CHECK:           %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_7]] : index
! CHECK:           cf.cond_br %[[VAL_17]], ^bb2(%[[VAL_7]], %[[VAL_8]] : index, index), ^bb5
! CHECK:         ^bb2(%[[VAL_18:.*]]: index, %[[VAL_19:.*]]: index):
! CHECK:           %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_7]] : index
! CHECK:           cf.cond_br %[[VAL_20]], ^bb3, ^bb4
! CHECK:         ^bb3:
! CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_18]], %[[VAL_5]] : index
! CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_15]], %[[VAL_5]] : index
! CHECK:           %[[VAL_23:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_14]]) %[[VAL_21]], %[[VAL_22]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_23]] : !fir.ref<f32>
! CHECK:           %[[VAL_24:.*]] = arith.subi %[[VAL_19]], %[[VAL_5]] : index
! CHECK:           cf.br ^bb2(%[[VAL_21]], %[[VAL_24]] : index, index)
! CHECK:         ^bb4:
! CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_15]], %[[VAL_5]] : index
! CHECK:           %[[VAL_26:.*]] = arith.subi %[[VAL_16]], %[[VAL_5]] : index
! CHECK:           cf.br ^bb1(%[[VAL_25]], %[[VAL_26]] : index, index)
! CHECK:         ^bb5:
! CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_10]], %[[VAL_0]] : index
! CHECK:           %[[VAL_28:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_7]] : index
! CHECK:           %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_27]], %[[VAL_7]] : index
! CHECK:           %[[VAL_30:.*]] = fir.slice %[[VAL_5]], %[[VAL_8]], %[[VAL_5]], %[[VAL_8]], %[[VAL_10]], %[[VAL_5]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           %[[VAL_31:.*]] = fir.allocmem !fir.array<2x?xf32>, %[[VAL_12]]
! CHECK:           cf.br ^bb6(%[[VAL_7]], %[[VAL_12]] : index, index)
! CHECK:         ^bb6(%[[VAL_32:.*]]: index, %[[VAL_33:.*]]: index):
! CHECK:           %[[VAL_34:.*]] = arith.cmpi sgt, %[[VAL_33]], %[[VAL_7]] : index
! CHECK:           cf.cond_br %[[VAL_34]], ^bb7(%[[VAL_7]], %[[VAL_8]] : index, index), ^bb10
! CHECK:         ^bb7(%[[VAL_35:.*]]: index, %[[VAL_36:.*]]: index):
! CHECK:           %[[VAL_37:.*]] = arith.cmpi sgt, %[[VAL_36]], %[[VAL_7]] : index
! CHECK:           cf.cond_br %[[VAL_37]], ^bb8, ^bb9
! CHECK:         ^bb8:
! CHECK:           %[[VAL_38:.*]] = arith.addi %[[VAL_35]], %[[VAL_5]] : index
! CHECK:           %[[VAL_39:.*]] = arith.addi %[[VAL_32]], %[[VAL_5]] : index
! CHECK:           %[[VAL_40:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_14]]) %[[VAL_38]], %[[VAL_39]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_41:.*]] = fir.array_coor %[[VAL_31]](%[[VAL_14]]) %[[VAL_38]], %[[VAL_39]] : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_42:.*]] = fir.load %[[VAL_40]] : !fir.ref<f32>
! CHECK:           fir.store %[[VAL_42]] to %[[VAL_41]] : !fir.ref<f32>
! CHECK:           %[[VAL_43:.*]] = arith.subi %[[VAL_36]], %[[VAL_5]] : index
! CHECK:           cf.br ^bb7(%[[VAL_38]], %[[VAL_43]] : index, index)
! CHECK:         ^bb9:
! CHECK:           %[[VAL_44:.*]] = arith.addi %[[VAL_32]], %[[VAL_5]] : index
! CHECK:           %[[VAL_45:.*]] = arith.subi %[[VAL_33]], %[[VAL_5]] : index
! CHECK:           cf.br ^bb6(%[[VAL_44]], %[[VAL_45]] : index, index)
! CHECK:         ^bb10:
! CHECK:           %[[VAL_46:.*]] = arith.subi %[[VAL_9]], %[[VAL_4]] : i32
! CHECK:           %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i32) -> index
! CHECK:           %[[VAL_48:.*]] = fir.slice %[[VAL_5]], %[[VAL_8]], %[[VAL_5]], %[[VAL_5]], %[[VAL_47]], %[[VAL_5]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           cf.br ^bb11(%[[VAL_7]], %[[VAL_29]] : index, index)
! CHECK:         ^bb11(%[[VAL_49:.*]]: index, %[[VAL_50:.*]]: index):
! CHECK:           %[[VAL_51:.*]] = arith.cmpi sgt, %[[VAL_50]], %[[VAL_7]] : index
! CHECK:           cf.cond_br %[[VAL_51]], ^bb12(%[[VAL_7]], %[[VAL_8]] : index, index), ^bb15(%[[VAL_7]], %[[VAL_12]] : index, index)
! CHECK:         ^bb12(%[[VAL_52:.*]]: index, %[[VAL_53:.*]]: index):
! CHECK:           %[[VAL_54:.*]] = arith.cmpi sgt, %[[VAL_53]], %[[VAL_7]] : index
! CHECK:           cf.cond_br %[[VAL_54]], ^bb13, ^bb14
! CHECK:         ^bb13:
! CHECK:           %[[VAL_55:.*]] = arith.addi %[[VAL_52]], %[[VAL_5]] : index
! CHECK:           %[[VAL_56:.*]] = arith.addi %[[VAL_49]], %[[VAL_5]] : index
! CHECK:           %[[VAL_57:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_14]]) {{\[}}%[[VAL_48]]] %[[VAL_55]], %[[VAL_56]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_58:.*]] = fir.load %[[VAL_57]] : !fir.ref<f32>
! CHECK:           %[[VAL_59:.*]] = arith.addf %[[VAL_58]], %[[VAL_3]] fastmath<contract> : f32
! CHECK:           %[[VAL_60:.*]] = fir.array_coor %[[VAL_31]](%[[VAL_14]]) {{\[}}%[[VAL_30]]] %[[VAL_55]], %[[VAL_56]] : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[VAL_59]] to %[[VAL_60]] : !fir.ref<f32>
! CHECK:           %[[VAL_61:.*]] = arith.subi %[[VAL_53]], %[[VAL_5]] : index
! CHECK:           cf.br ^bb12(%[[VAL_55]], %[[VAL_61]] : index, index)
! CHECK:         ^bb14:
! CHECK:           %[[VAL_62:.*]] = arith.addi %[[VAL_49]], %[[VAL_5]] : index
! CHECK:           %[[VAL_63:.*]] = arith.subi %[[VAL_50]], %[[VAL_5]] : index
! CHECK:           cf.br ^bb11(%[[VAL_62]], %[[VAL_63]] : index, index)
! CHECK:         ^bb15(%[[VAL_64:.*]]: index, %[[VAL_65:.*]]: index):
! CHECK:           %[[VAL_66:.*]] = arith.cmpi sgt, %[[VAL_65]], %[[VAL_7]] : index
! CHECK:           cf.cond_br %[[VAL_66]], ^bb16(%[[VAL_7]], %[[VAL_8]] : index, index), ^bb19
! CHECK:         ^bb16(%[[VAL_67:.*]]: index, %[[VAL_68:.*]]: index):
! CHECK:           %[[VAL_69:.*]] = arith.cmpi sgt, %[[VAL_68]], %[[VAL_7]] : index
! CHECK:           cf.cond_br %[[VAL_69]], ^bb17, ^bb18
! CHECK:         ^bb17:
! CHECK:           %[[VAL_70:.*]] = arith.addi %[[VAL_67]], %[[VAL_5]] : index
! CHECK:           %[[VAL_71:.*]] = arith.addi %[[VAL_64]], %[[VAL_5]] : index
! CHECK:           %[[VAL_72:.*]] = fir.array_coor %[[VAL_31]](%[[VAL_14]]) %[[VAL_70]], %[[VAL_71]] : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_73:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_14]]) %[[VAL_70]], %[[VAL_71]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_74:.*]] = fir.load %[[VAL_72]] : !fir.ref<f32>
! CHECK:           fir.store %[[VAL_74]] to %[[VAL_73]] : !fir.ref<f32>
! CHECK:           %[[VAL_75:.*]] = arith.subi %[[VAL_68]], %[[VAL_5]] : index
! CHECK:           cf.br ^bb16(%[[VAL_70]], %[[VAL_75]] : index, index)
! CHECK:         ^bb18:
! CHECK:           %[[VAL_76:.*]] = arith.addi %[[VAL_64]], %[[VAL_5]] : index
! CHECK:           %[[VAL_77:.*]] = arith.subi %[[VAL_65]], %[[VAL_5]] : index
! CHECK:           cf.br ^bb15(%[[VAL_76]], %[[VAL_77]] : index, index)
! CHECK:         ^bb19:
! CHECK:           fir.freemem %[[VAL_31]] : !fir.heap<!fir.array<2x?xf32>>
! CHECK:           %[[VAL_78:.*]] = fir.address_of(@_QQclX88e38683fcd4166fc29b4ce91ec570e4) : !fir.ref<!fir.char<1,80>>
! CHECK:           %[[VAL_79:.*]] = fir.convert %[[VAL_78]] : (!fir.ref<!fir.char<1,80>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_80:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_2]], %[[VAL_79]], %[[VAL_1]]) fastmath<contract> {fir.llvm_memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, llvm.nocallback, llvm.nosync} : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_81:.*]] = fir.slice %[[VAL_5]], %[[VAL_8]], %[[VAL_5]], %[[VAL_5]], %[[VAL_8]], %[[VAL_5]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           %[[VAL_82:.*]] = fir.embox %[[VAL_13]](%[[VAL_14]]) {{\[}}%[[VAL_81]]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x2xf32>>
! CHECK:           %[[VAL_83:.*]] = fir.convert %[[VAL_82]] : (!fir.box<!fir.array<?x2xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_84:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_80]], %[[VAL_83]]) fastmath<contract> {llvm.nocallback, llvm.nosync} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_85:.*]] = fir.slice %[[VAL_5]], %[[VAL_8]], %[[VAL_5]], %[[VAL_47]], %[[VAL_10]], %[[VAL_5]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           %[[VAL_86:.*]] = fir.embox %[[VAL_13]](%[[VAL_14]]) {{\[}}%[[VAL_85]]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:           %[[VAL_87:.*]] = fir.convert %[[VAL_86]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_88:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_80]], %[[VAL_87]]) fastmath<contract> {llvm.nocallback, llvm.nosync} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_89:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_80]]) fastmath<contract> {fir.llvm_memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, llvm.nocallback, llvm.nosync} : (!fir.ref<i8>) -> i32
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPss4(
! CHECK-SAME:                      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_0:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_3:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_4:.*]] = arith.constant 7.000000e+00 : f32
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_7:.*]] = arith.constant -2.000000e+00 : f32
! CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_9:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_8]] : index
! CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_8]] : index
! CHECK:           %[[VAL_13:.*]] = fir.alloca !fir.array<?x2xf32>, %[[VAL_12]] {bindc_name = "aa", uniq_name = "_QFss4Eaa"}
! CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_12]], %[[VAL_1]] : (index, index) -> !fir.shape<2>
! CHECK:           cf.br ^bb1(%[[VAL_8]], %[[VAL_1]] : index, index)
! CHECK:         ^bb1(%[[VAL_15:.*]]: index, %[[VAL_16:.*]]: index):
! CHECK:           %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_8]] : index
! CHECK:           cf.cond_br %[[VAL_17]], ^bb2(%[[VAL_8]], %[[VAL_12]] : index, index), ^bb5
! CHECK:         ^bb2(%[[VAL_18:.*]]: index, %[[VAL_19:.*]]: index):
! CHECK:           %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_8]] : index
! CHECK:           cf.cond_br %[[VAL_20]], ^bb3, ^bb4
! CHECK:         ^bb3:
! CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_18]], %[[VAL_6]] : index
! CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_15]], %[[VAL_6]] : index
! CHECK:           %[[VAL_23:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_14]]) %[[VAL_21]], %[[VAL_22]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_23]] : !fir.ref<f32>
! CHECK:           %[[VAL_24:.*]] = arith.subi %[[VAL_19]], %[[VAL_6]] : index
! CHECK:           cf.br ^bb2(%[[VAL_21]], %[[VAL_24]] : index, index)
! CHECK:         ^bb4:
! CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_15]], %[[VAL_6]] : index
! CHECK:           %[[VAL_26:.*]] = arith.subi %[[VAL_16]], %[[VAL_6]] : index
! CHECK:           cf.br ^bb1(%[[VAL_25]], %[[VAL_26]] : index, index)
! CHECK:         ^bb5:
! CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_10]], %[[VAL_0]] : index
! CHECK:           %[[VAL_28:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_8]] : index
! CHECK:           %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_27]], %[[VAL_8]] : index
! CHECK:           %[[VAL_30:.*]] = fir.slice %[[VAL_1]], %[[VAL_10]], %[[VAL_6]], %[[VAL_6]], %[[VAL_1]], %[[VAL_6]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           %[[VAL_31:.*]] = fir.allocmem !fir.array<?x2xf32>, %[[VAL_12]]
! CHECK:           cf.br ^bb6(%[[VAL_8]], %[[VAL_1]] : index, index)
! CHECK:         ^bb6(%[[VAL_32:.*]]: index, %[[VAL_33:.*]]: index):
! CHECK:           %[[VAL_34:.*]] = arith.cmpi sgt, %[[VAL_33]], %[[VAL_8]] : index
! CHECK:           cf.cond_br %[[VAL_34]], ^bb7(%[[VAL_8]], %[[VAL_12]] : index, index), ^bb10
! CHECK:         ^bb7(%[[VAL_35:.*]]: index, %[[VAL_36:.*]]: index):
! CHECK:           %[[VAL_37:.*]] = arith.cmpi sgt, %[[VAL_36]], %[[VAL_8]] : index
! CHECK:           cf.cond_br %[[VAL_37]], ^bb8, ^bb9
! CHECK:         ^bb8:
! CHECK:           %[[VAL_38:.*]] = arith.addi %[[VAL_35]], %[[VAL_6]] : index
! CHECK:           %[[VAL_39:.*]] = arith.addi %[[VAL_32]], %[[VAL_6]] : index
! CHECK:           %[[VAL_40:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_14]]) %[[VAL_38]], %[[VAL_39]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_41:.*]] = fir.array_coor %[[VAL_31]](%[[VAL_14]]) %[[VAL_38]], %[[VAL_39]] : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_42:.*]] = fir.load %[[VAL_40]] : !fir.ref<f32>
! CHECK:           fir.store %[[VAL_42]] to %[[VAL_41]] : !fir.ref<f32>
! CHECK:           %[[VAL_43:.*]] = arith.subi %[[VAL_36]], %[[VAL_6]] : index
! CHECK:           cf.br ^bb7(%[[VAL_38]], %[[VAL_43]] : index, index)
! CHECK:         ^bb9:
! CHECK:           %[[VAL_44:.*]] = arith.addi %[[VAL_32]], %[[VAL_6]] : index
! CHECK:           %[[VAL_45:.*]] = arith.subi %[[VAL_33]], %[[VAL_6]] : index
! CHECK:           cf.br ^bb6(%[[VAL_44]], %[[VAL_45]] : index, index)
! CHECK:         ^bb10:
! CHECK:           %[[VAL_46:.*]] = arith.subi %[[VAL_9]], %[[VAL_5]] : i32
! CHECK:           %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i32) -> index
! CHECK:           %[[VAL_48:.*]] = fir.slice %[[VAL_6]], %[[VAL_47]], %[[VAL_6]], %[[VAL_6]], %[[VAL_1]], %[[VAL_6]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           cf.br ^bb11(%[[VAL_8]], %[[VAL_1]] : index, index)
! CHECK:         ^bb11(%[[VAL_49:.*]]: index, %[[VAL_50:.*]]: index):
! CHECK:           %[[VAL_51:.*]] = arith.cmpi sgt, %[[VAL_50]], %[[VAL_8]] : index
! CHECK:           cf.cond_br %[[VAL_51]], ^bb12(%[[VAL_8]], %[[VAL_29]] : index, index), ^bb15(%[[VAL_8]], %[[VAL_1]] : index, index)
! CHECK:         ^bb12(%[[VAL_52:.*]]: index, %[[VAL_53:.*]]: index):
! CHECK:           %[[VAL_54:.*]] = arith.cmpi sgt, %[[VAL_53]], %[[VAL_8]] : index
! CHECK:           cf.cond_br %[[VAL_54]], ^bb13, ^bb14
! CHECK:         ^bb13:
! CHECK:           %[[VAL_55:.*]] = arith.addi %[[VAL_52]], %[[VAL_6]] : index
! CHECK:           %[[VAL_56:.*]] = arith.addi %[[VAL_49]], %[[VAL_6]] : index
! CHECK:           %[[VAL_57:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_14]]) {{\[}}%[[VAL_48]]] %[[VAL_55]], %[[VAL_56]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_58:.*]] = fir.load %[[VAL_57]] : !fir.ref<f32>
! CHECK:           %[[VAL_59:.*]] = arith.addf %[[VAL_58]], %[[VAL_4]] fastmath<contract> : f32
! CHECK:           %[[VAL_60:.*]] = fir.array_coor %[[VAL_31]](%[[VAL_14]]) {{\[}}%[[VAL_30]]] %[[VAL_55]], %[[VAL_56]] : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[VAL_59]] to %[[VAL_60]] : !fir.ref<f32>
! CHECK:           %[[VAL_61:.*]] = arith.subi %[[VAL_53]], %[[VAL_6]] : index
! CHECK:           cf.br ^bb12(%[[VAL_55]], %[[VAL_61]] : index, index)
! CHECK:         ^bb14:
! CHECK:           %[[VAL_62:.*]] = arith.addi %[[VAL_49]], %[[VAL_6]] : index
! CHECK:           %[[VAL_63:.*]] = arith.subi %[[VAL_50]], %[[VAL_6]] : index
! CHECK:           cf.br ^bb11(%[[VAL_62]], %[[VAL_63]] : index, index)
! CHECK:         ^bb15(%[[VAL_64:.*]]: index, %[[VAL_65:.*]]: index):
! CHECK:           %[[VAL_66:.*]] = arith.cmpi sgt, %[[VAL_65]], %[[VAL_8]] : index
! CHECK:           cf.cond_br %[[VAL_66]], ^bb16(%[[VAL_8]], %[[VAL_12]] : index, index), ^bb19
! CHECK:         ^bb16(%[[VAL_67:.*]]: index, %[[VAL_68:.*]]: index):
! CHECK:           %[[VAL_69:.*]] = arith.cmpi sgt, %[[VAL_68]], %[[VAL_8]] : index
! CHECK:           cf.cond_br %[[VAL_69]], ^bb17, ^bb18
! CHECK:         ^bb17:
! CHECK:           %[[VAL_70:.*]] = arith.addi %[[VAL_67]], %[[VAL_6]] : index
! CHECK:           %[[VAL_71:.*]] = arith.addi %[[VAL_64]], %[[VAL_6]] : index
! CHECK:           %[[VAL_72:.*]] = fir.array_coor %[[VAL_31]](%[[VAL_14]]) %[[VAL_70]], %[[VAL_71]] : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_73:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_14]]) %[[VAL_70]], %[[VAL_71]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_74:.*]] = fir.load %[[VAL_72]] : !fir.ref<f32>
! CHECK:           fir.store %[[VAL_74]] to %[[VAL_73]] : !fir.ref<f32>
! CHECK:           %[[VAL_75:.*]] = arith.subi %[[VAL_68]], %[[VAL_6]] : index
! CHECK:           cf.br ^bb16(%[[VAL_70]], %[[VAL_75]] : index, index)
! CHECK:         ^bb18:
! CHECK:           %[[VAL_76:.*]] = arith.addi %[[VAL_64]], %[[VAL_6]] : index
! CHECK:           %[[VAL_77:.*]] = arith.subi %[[VAL_65]], %[[VAL_6]] : index
! CHECK:           cf.br ^bb15(%[[VAL_76]], %[[VAL_77]] : index, index)
! CHECK:         ^bb19:
! CHECK:           fir.freemem %[[VAL_31]] : !fir.heap<!fir.array<?x2xf32>>
! CHECK:           %[[VAL_78:.*]] = fir.address_of(@_QQclX88e38683fcd4166fc29b4ce91ec570e4) : !fir.ref<!fir.char<1,80>>
! CHECK:           %[[VAL_79:.*]] = fir.convert %[[VAL_78]] : (!fir.ref<!fir.char<1,80>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_80:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_3]], %[[VAL_79]], %[[VAL_2]]) fastmath<contract> {fir.llvm_memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, llvm.nocallback, llvm.nosync} : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_81:.*]] = fir.slice %[[VAL_6]], %[[VAL_1]], %[[VAL_6]], %[[VAL_6]], %[[VAL_1]], %[[VAL_6]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           %[[VAL_82:.*]] = fir.embox %[[VAL_13]](%[[VAL_14]]) {{\[}}%[[VAL_81]]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<2x?xf32>>
! CHECK:           %[[VAL_83:.*]] = fir.convert %[[VAL_82]] : (!fir.box<!fir.array<2x?xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_84:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_80]], %[[VAL_83]]) fastmath<contract> {llvm.nocallback, llvm.nosync} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_85:.*]] = fir.slice %[[VAL_47]], %[[VAL_10]], %[[VAL_6]], %[[VAL_6]], %[[VAL_1]], %[[VAL_6]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           %[[VAL_86:.*]] = fir.embox %[[VAL_13]](%[[VAL_14]]) {{\[}}%[[VAL_85]]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:           %[[VAL_87:.*]] = fir.convert %[[VAL_86]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_88:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_80]], %[[VAL_87]]) fastmath<contract> {llvm.nocallback, llvm.nosync} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_89:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_80]]) fastmath<contract> {fir.llvm_memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, llvm.nocallback, llvm.nosync} : (!fir.ref<i8>) -> i32
! CHECK:           return
! CHECK:         }

subroutine ss2(N)
  real aa(N)
  aa = -2
  aa(2:N) = aa(1:N-1) + 7.0
  print*, aa(1:2), aa(N-1:N)
end

subroutine ss3(N)
  real aa(2,N)
  aa = -2
  aa(:,2:N) = aa(:,1:N-1) + 7.0
  print*, aa(:,1:2), aa(:,N-1:N)
end

subroutine ss4(N)
  real aa(N,2)
  aa = -2
  aa(2:N,:) = aa(1:N-1,:) + 7.0
  print*, aa(1:2,:), aa(N-1:N,:)
end


! CHECK-LABEL: func @_QPtt1
subroutine tt1
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: %[[temp3:[0-9]+]] = fir.allocmem !fir.array<3xf32>
  ! CHECK: br ^bb1(%[[temp3]]
  ! CHECK-NEXT: ^bb1(%[[temp3arg:[0-9]+]]: !fir.heap<!fir.array<3xf32>>
  ! CHECK: %[[temp1:[0-9]+]] = fir.allocmem !fir.array<1xf32>
  ! CHECK: fir.call @_QFtt1Pr
  ! CHECK: fir.call @realloc
  ! CHECK: fir.freemem %[[temp1]] : !fir.heap<!fir.array<1xf32>>
  ! CHECK: %[[temp3x:[0-9]+]] = fir.allocmem !fir.array<3xf32>
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  ! CHECK-NEXT: fir.freemem %[[temp3x]] : !fir.heap<!fir.array<3xf32>>
  ! CHECK-NEXT: fir.freemem %[[temp3arg]] : !fir.heap<!fir.array<3xf32>>
  ! CHECK-NEXT: fir.call @_FortranAioEndIoStatement
  print*, [(r([7.0]),i=1,3)]
contains
  ! CHECK-LABEL: func private @_QFtt1Pr
  function r(x)
    real x(:)
    r = x(1)
  end
end
