! Test disable identical block merge in the canonicalizer pass in bbc.
! Temporary fix for issue #1021.
! RUN: bbc -hlfir=false %s -o - | FileCheck %s

MODULE DMUMPS_SOL_LR
IMPLICIT NONE

TYPE BLR_STRUC_T
  INTEGER, DIMENSION(:), POINTER  :: PANELS_L 
  INTEGER, DIMENSION(:), POINTER  :: PANELS_U
  INTEGER, DIMENSION(:), POINTER :: BEGS_BLR_STATIC
END TYPE BLR_STRUC_T

TYPE(BLR_STRUC_T), POINTER, DIMENSION(:), SAVE :: BLR_ARRAY

CONTAINS

SUBROUTINE DMUMPS_SOL_FWD_LR_SU( IWHDLR, MTYPE )

  INTEGER, INTENT(IN) :: IWHDLR, MTYPE
  INTEGER :: NPARTSASS, NB_BLR

  IF (MTYPE.EQ.1) THEN
    IF ( associated( BLR_ARRAY(IWHDLR)%PANELS_L ) ) THEN
      NPARTSASS = size( BLR_ARRAY(IWHDLR)%PANELS_L )
      NB_BLR = size( BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC ) - 1
    ENDIF
  ELSE
    IF ( associated( BLR_ARRAY(IWHDLR)%PANELS_U ) ) THEN
      NPARTSASS = size( BLR_ARRAY(IWHDLR)%PANELS_U )
      NB_BLR = size( BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC ) - 1
    ENDIF
  ENDIF

END SUBROUTINE DMUMPS_SOL_FWD_LR_SU 

END MODULE DMUMPS_SOL_LR

! CHECK-LABEL:   func.func @_QMdmumps_sol_lrPdmumps_sol_fwd_lr_su(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "iwhdlr"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "mtype"}) {
! CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_3:.*]] = fir.address_of(@_QMdmumps_sol_lrEblr_array) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "nb_blr", uniq_name = "_QMdmumps_sol_lrFdmumps_sol_fwd_lr_suEnb_blr"}
! CHECK:           %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "npartsass", uniq_name = "_QMdmumps_sol_lrFdmumps_sol_fwd_lr_suEnpartsass"}
! CHECK:           %[[VAL_6:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_2]] : i32
! CHECK:           cf.cond_br %[[VAL_7]], ^bb1, ^bb3
! CHECK:         ^bb1:
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:           %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_10:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> i64
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_9]]#0 : (index) -> i64
! CHECK:           %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:           %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_13]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, i64) -> !fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:           %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_14]], panels_l : (!fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_17:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_19:.*]] = arith.cmpi ne, %[[VAL_18]], %[[VAL_0]] : i64
! CHECK:           cf.cond_br %[[VAL_19]], ^bb2, ^bb5
! CHECK:         ^bb2:
! CHECK:           %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_16]], %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]]#1 : (index) -> i32
! CHECK:           fir.store %[[VAL_21]] to %[[VAL_5]] : !fir.ref<i32>
! CHECK:           %[[VAL_22:.*]] = fir.coordinate_of %[[VAL_14]], begs_blr_static : (!fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_24:.*]]:3 = fir.box_dims %[[VAL_23]], %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]]#1 : (index) -> i32
! CHECK:           %[[VAL_26:.*]] = arith.subi %[[VAL_25]], %[[VAL_2]] : i32
! CHECK:           fir.store %[[VAL_26]] to %[[VAL_4]] : !fir.ref<i32>
! CHECK:           cf.br ^bb5
! CHECK:         ^bb3:
! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:           %[[VAL_28:.*]]:3 = fir.box_dims %[[VAL_27]], %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_29:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_28]]#0 : (index) -> i64
! CHECK:           %[[VAL_32:.*]] = arith.subi %[[VAL_30]], %[[VAL_31]] : i64
! CHECK:           %[[VAL_33:.*]] = fir.coordinate_of %[[VAL_27]], %[[VAL_32]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, i64) -> !fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:           %[[VAL_34:.*]] = fir.coordinate_of %[[VAL_33]], panels_u : (!fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_35:.*]] = fir.load %[[VAL_34]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_36:.*]] = fir.box_addr %[[VAL_35]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_38:.*]] = arith.cmpi ne, %[[VAL_37]], %[[VAL_0]] : i64
! CHECK:           cf.cond_br %[[VAL_38]], ^bb4, ^bb5
! CHECK:         ^bb4:
! CHECK:           %[[VAL_39:.*]]:3 = fir.box_dims %[[VAL_35]], %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_39]]#1 : (index) -> i32
! CHECK:           fir.store %[[VAL_40]] to %[[VAL_5]] : !fir.ref<i32>
! CHECK:           %[[VAL_41:.*]] = fir.coordinate_of %[[VAL_33]], begs_blr_static : (!fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_42:.*]] = fir.load %[[VAL_41]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_43:.*]]:3 = fir.box_dims %[[VAL_42]], %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_44:.*]] = fir.convert %[[VAL_43]]#1 : (index) -> i32
! CHECK:           %[[VAL_45:.*]] = arith.subi %[[VAL_44]], %[[VAL_2]] : i32
! CHECK:           fir.store %[[VAL_45]] to %[[VAL_4]] : !fir.ref<i32>
! CHECK:           cf.br ^bb5
! CHECK:         ^bb5:
! CHECK:           return
! CHECK:         }
