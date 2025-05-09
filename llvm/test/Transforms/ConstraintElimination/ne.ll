; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --check-attributes
; RUN: opt -passes=constraint-elimination -S %s | FileCheck %s

declare void @llvm.assume(i1)

define i1 @test_eq_ne_0(i8 %a, i8 %b) {
; CHECK-LABEL: @test_eq_ne_0(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[A:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[C_2:%.*]] = icmp ne i8 [[A]], [[B:%.*]]
; CHECK-NEXT:    [[RES_1:%.*]] = xor i1 false, true
; CHECK-NEXT:    [[RES_2:%.*]] = xor i1 [[RES_1]], [[C_2]]
; CHECK-NEXT:    ret i1 [[RES_2]]
; CHECK:       else:
; CHECK-NEXT:    [[C_3:%.*]] = icmp ne i8 [[A]], 1
; CHECK-NEXT:    [[C_4:%.*]] = icmp ne i8 [[A]], [[B]]
; CHECK-NEXT:    [[RES_3:%.*]] = xor i1 true, [[C_3]]
; CHECK-NEXT:    [[RES_4:%.*]] = xor i1 [[RES_3]], [[C_4]]
; CHECK-NEXT:    ret i1 [[RES_4]]
;
entry:
  %cmp = icmp eq i8 %a, 0
  br i1 %cmp, label %then, label %else

then:
  %f.1 = icmp ne i8 %a, 0
  %c.1 = icmp ne i8 %a, 1
  %c.2 = icmp ne i8 %a, %b
  %res.1 = xor i1 %f.1, %c.1
  %res.2 = xor i1 %res.1, %c.2
  ret i1 %res.2

else:
  %t.1 = icmp ne i8 %a, 0
  %c.3 = icmp ne i8 %a, 1
  %c.4 = icmp ne i8 %a, %b
  %res.3 = xor i1 %t.1, %c.3
  %res.4 = xor i1 %res.3, %c.4
  ret i1 %res.4
}

define i1 @test_ne_eq_0(i8 %a, i8 %b) {
; CHECK-LABEL: @test_ne_eq_0(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp ne i8 [[A:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[C_1:%.*]] = icmp ne i8 [[A]], 1
; CHECK-NEXT:    [[RES_1:%.*]] = xor i1 true, [[C_1]]
; CHECK-NEXT:    [[C_2:%.*]] = icmp ne i8 [[A]], [[B:%.*]]
; CHECK-NEXT:    [[RES_2:%.*]] = xor i1 [[RES_1]], [[C_2]]
; CHECK-NEXT:    [[C_3:%.*]] = icmp eq i8 [[A]], [[B]]
; CHECK-NEXT:    [[RES_3:%.*]] = xor i1 [[RES_2]], [[C_3]]
; CHECK-NEXT:    [[RES_4:%.*]] = xor i1 [[RES_3]], false
; CHECK-NEXT:    [[RES_5:%.*]] = xor i1 [[RES_4]], true
; CHECK-NEXT:    [[RES_6:%.*]] = xor i1 [[RES_5]], true
; CHECK-NEXT:    [[C_5:%.*]] = icmp ugt i8 [[A]], 1
; CHECK-NEXT:    [[RES_7:%.*]] = xor i1 [[RES_6]], [[C_5]]
; CHECK-NEXT:    [[C_6:%.*]] = icmp sgt i8 [[A]], 0
; CHECK-NEXT:    [[RES_8:%.*]] = xor i1 [[RES_7]], [[C_6]]
; CHECK-NEXT:    ret i1 [[RES_8]]
; CHECK:       else:
; CHECK-NEXT:    [[RES_9:%.*]] = xor i1 false, true
; CHECK-NEXT:    [[C_8:%.*]] = icmp ne i8 [[A]], [[B]]
; CHECK-NEXT:    [[RES_10:%.*]] = xor i1 [[RES_9]], [[C_8]]
; CHECK-NEXT:    [[C_9:%.*]] = icmp eq i8 [[A]], [[B]]
; CHECK-NEXT:    [[RES_11:%.*]] = xor i1 [[RES_10]], [[C_9]]
; CHECK-NEXT:    [[RES_12:%.*]] = xor i1 [[RES_11]], true
; CHECK-NEXT:    [[RES_13:%.*]] = xor i1 [[RES_12]], false
; CHECK-NEXT:    [[RES_14:%.*]] = xor i1 [[RES_13]], false
; CHECK-NEXT:    [[RES_15:%.*]] = xor i1 [[RES_14]], false
; CHECK-NEXT:    [[RES_16:%.*]] = xor i1 [[RES_15]], false
; CHECK-NEXT:    ret i1 [[RES_16]]
;
entry:
  %cmp = icmp ne i8 %a, 0
  br i1 %cmp, label %then, label %else

then:
  %t.1 = icmp ne i8 %a, 0
  %c.1 = icmp ne i8 %a, 1
  %res.1 = xor i1 %t.1, %c.1

  %c.2 = icmp ne i8 %a, %b
  %res.2 = xor i1 %res.1, %c.2

  %c.3 = icmp eq i8 %a, %b
  %res.3 = xor i1 %res.2, %c.3

  %c.4 = icmp eq i8 %a, 0
  %res.4 = xor i1 %res.3, %c.4

  %t.2 = icmp ugt i8 %a, 0
  %res.5 = xor i1 %res.4, %t.2

  %t.3 = icmp uge i8 %a, 1
  %res.6 = xor i1 %res.5, %t.3

  %c.5 = icmp ugt i8 %a, 1
  %res.7 = xor i1 %res.6, %c.5

  %c.6 = icmp sgt i8 %a, 0
  %res.8 = xor i1 %res.7, %c.6

  ret i1 %res.8

else:
  %f.1 = icmp ne i8 %a, 0
  %c.7 = icmp ne i8 %a, 1
  %res.9 = xor i1 %f.1, %c.7

  %c.8 = icmp ne i8 %a, %b
  %res.10 = xor i1 %res.9, %c.8

  %c.9 = icmp eq i8 %a, %b
  %res.11 = xor i1 %res.10, %c.9

  %c.10 = icmp eq i8 %a, 0
  %res.12 = xor i1 %res.11, %c.10

  %f.2 = icmp ugt i8 %a, 0
  %res.13 = xor i1 %res.12, %f.2

  %f.3 = icmp uge i8 %a, 1
  %res.14 = xor i1 %res.13, %f.3

  %c.11 = icmp ugt i8 %a, 1
  %res.15 = xor i1 %res.14, %c.11

  %c.12 = icmp sgt i8 %a, 0
  %res.16 = xor i1 %res.15, %c.12

  ret i1 %res.16
}

define i1 @test_eq_ne_1(i8 %a, i8 %b) {
; CHECK-LABEL: @test_eq_ne_1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[A:%.*]], 1
; CHECK-NEXT:    br i1 [[CMP]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[C_2:%.*]] = icmp ne i8 [[A]], [[B:%.*]]
; CHECK-NEXT:    [[RES_1:%.*]] = xor i1 true, false
; CHECK-NEXT:    [[RES_2:%.*]] = xor i1 [[RES_1]], [[C_2]]
; CHECK-NEXT:    ret i1 [[RES_2]]
; CHECK:       else:
; CHECK-NEXT:    [[T_1:%.*]] = icmp ne i8 [[A]], 0
; CHECK-NEXT:    [[C_3:%.*]] = icmp ne i8 [[A]], 1
; CHECK-NEXT:    [[C_4:%.*]] = icmp ne i8 [[A]], [[B]]
; CHECK-NEXT:    [[RES_3:%.*]] = xor i1 [[T_1]], [[C_3]]
; CHECK-NEXT:    [[RES_4:%.*]] = xor i1 [[RES_3]], [[C_4]]
; CHECK-NEXT:    ret i1 [[RES_4]]
;
entry:
  %cmp = icmp eq i8 %a, 1
  br i1 %cmp, label %then, label %else

then:
  %f.1 = icmp ne i8 %a, 0
  %c.1 = icmp ne i8 %a, 1
  %c.2 = icmp ne i8 %a, %b
  %res.1 = xor i1 %f.1, %c.1
  %res.2 = xor i1 %res.1, %c.2
  ret i1 %res.2

else:
  %t.1 = icmp ne i8 %a, 0
  %c.3 = icmp ne i8 %a, 1
  %c.4 = icmp ne i8 %a, %b
  %res.3 = xor i1 %t.1, %c.3
  %res.4 = xor i1 %res.3, %c.4
  ret i1 %res.4
}

define i1 @test_ne_eq_1(i8 %a, i8 %b) {
; CHECK-LABEL: @test_ne_eq_1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp ne i8 [[A:%.*]], 1
; CHECK-NEXT:    br i1 [[CMP]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[T_1:%.*]] = icmp ne i8 [[A]], 1
; CHECK-NEXT:    [[C_1:%.*]] = icmp ne i8 [[A]], 0
; CHECK-NEXT:    [[RES_1:%.*]] = xor i1 [[T_1]], [[C_1]]
; CHECK-NEXT:    [[C_2:%.*]] = icmp ne i8 [[A]], [[B:%.*]]
; CHECK-NEXT:    [[RES_2:%.*]] = xor i1 [[RES_1]], [[C_2]]
; CHECK-NEXT:    [[C_3:%.*]] = icmp eq i8 [[A]], [[B]]
; CHECK-NEXT:    [[RES_3:%.*]] = xor i1 [[RES_2]], [[C_3]]
; CHECK-NEXT:    [[C_4:%.*]] = icmp eq i8 [[A]], 0
; CHECK-NEXT:    [[RES_4:%.*]] = xor i1 [[RES_3]], [[C_4]]
; CHECK-NEXT:    [[C_5:%.*]] = icmp ugt i8 [[A]], 0
; CHECK-NEXT:    [[RES_5:%.*]] = xor i1 [[RES_4]], [[C_5]]
; CHECK-NEXT:    [[C_6:%.*]] = icmp uge i8 [[A]], 1
; CHECK-NEXT:    [[RES_6:%.*]] = xor i1 [[RES_5]], [[C_6]]
; CHECK-NEXT:    [[C_7:%.*]] = icmp ugt i8 [[A]], 1
; CHECK-NEXT:    [[RES_7:%.*]] = xor i1 [[RES_6]], [[C_5]]
; CHECK-NEXT:    [[C_8:%.*]] = icmp sgt i8 [[A]], 0
; CHECK-NEXT:    [[RES_8:%.*]] = xor i1 [[RES_7]], [[C_6]]
; CHECK-NEXT:    ret i1 [[RES_8]]
; CHECK:       else:
; CHECK-NEXT:    [[RES_9:%.*]] = xor i1 true, false
; CHECK-NEXT:    [[C_10:%.*]] = icmp ne i8 [[A]], [[B]]
; CHECK-NEXT:    [[RES_10:%.*]] = xor i1 [[RES_9]], [[C_10]]
; CHECK-NEXT:    [[C_11:%.*]] = icmp eq i8 [[A]], [[B]]
; CHECK-NEXT:    [[RES_11:%.*]] = xor i1 [[RES_10]], [[C_11]]
; CHECK-NEXT:    [[RES_12:%.*]] = xor i1 [[RES_11]], false
; CHECK-NEXT:    [[RES_13:%.*]] = xor i1 [[RES_12]], true
; CHECK-NEXT:    [[RES_14:%.*]] = xor i1 [[RES_13]], true
; CHECK-NEXT:    [[RES_15:%.*]] = xor i1 [[RES_14]], false
; CHECK-NEXT:    [[RES_16:%.*]] = xor i1 [[RES_15]], true
; CHECK-NEXT:    ret i1 [[RES_16]]
;
entry:
  %cmp = icmp ne i8 %a, 1
  br i1 %cmp, label %then, label %else

then:
  %t.1 = icmp ne i8 %a, 1
  %c.1 = icmp ne i8 %a, 0
  %res.1 = xor i1 %t.1, %c.1

  %c.2 = icmp ne i8 %a, %b
  %res.2 = xor i1 %res.1, %c.2

  %c.3 = icmp eq i8 %a, %b
  %res.3 = xor i1 %res.2, %c.3

  %c.4 = icmp eq i8 %a, 0
  %res.4 = xor i1 %res.3, %c.4

  %c.5 = icmp ugt i8 %a, 0
  %res.5 = xor i1 %res.4, %c.5

  %c.6 = icmp uge i8 %a, 1
  %res.6 = xor i1 %res.5, %c.6

  %c.7 = icmp ugt i8 %a, 1
  %res.7 = xor i1 %res.6, %c.5

  %c.8 = icmp sgt i8 %a, 0
  %res.8 = xor i1 %res.7, %c.6

  ret i1 %res.8

else:
  %t.2 = icmp ne i8 %a, 0
  %c.9 = icmp ne i8 %a, 1
  %res.9 = xor i1 %t.2, %c.9

  %c.10 = icmp ne i8 %a, %b
  %res.10 = xor i1 %res.9, %c.10

  %c.11 = icmp eq i8 %a, %b
  %res.11 = xor i1 %res.10, %c.11

  %f.1 = icmp eq i8 %a, 0
  %res.12 = xor i1 %res.11, %f.1

  %t.3 = icmp ugt i8 %a, 0
  %res.13 = xor i1 %res.12, %t.3

  %t.4 = icmp uge i8 %a, 1
  %res.14 = xor i1 %res.13, %t.4

  %f.2 = icmp ugt i8 %a, 1
  %res.15 = xor i1 %res.14, %f.2

  %c.12 = icmp sgt i8 %a, 0
  %res.16 = xor i1 %res.15, %c.12

  ret i1 %res.16
}

define i1 @assume_b_plus_1_ult_a(i64 %a, i64 %b) {
; CHECK-LABEL: @assume_b_plus_1_ult_a(
; CHECK-NEXT:    [[TMP1:%.*]] = add nuw i64 [[B:%.*]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = icmp ult i64 [[TMP1]], [[A:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[TMP2]])
; CHECK-NEXT:    ret i1 true
;
  %1 = add nuw i64 %b, 1
  %2 = icmp ult i64 %1, %a
  tail call void @llvm.assume(i1 %2)
  %3 = icmp ne i64 %a, %b
  ret i1 %3
}

define i1 @assume_a_gt_b_and_b_ge_c(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: @assume_a_gt_b_and_b_ge_c(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ugt i64 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[TMP1]])
; CHECK-NEXT:    [[TMP2:%.*]] = icmp uge i64 [[B]], [[C:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[TMP2]])
; CHECK-NEXT:    ret i1 true
;
  %1 = icmp ugt i64 %a, %b
  tail call void @llvm.assume(i1 %1)
  %2 = icmp uge i64 %b, %c
  tail call void @llvm.assume(i1 %2)
  %3 = icmp ne i64 %a, %c
  ret i1 %3
}

define i1 @assume_a_ne_b_and_b_ne_c(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: @assume_a_ne_b_and_b_ne_c(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ne i64 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[TMP1]])
; CHECK-NEXT:    [[TMP2:%.*]] = icmp ne i64 [[B]], [[C:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[TMP2]])
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ne i64 [[A]], [[C]]
; CHECK-NEXT:    ret i1 [[TMP3]]
;
  %1 = icmp ne i64 %a, %b
  tail call void @llvm.assume(i1 %1)
  %2 = icmp ne i64 %b, %c
  tail call void @llvm.assume(i1 %2)
  %3 = icmp ne i64 %a, %c
  ret i1 %3
}

define i1 @assume_1a(i64 %a, i64 %b) {
; CHECK-LABEL: @assume_1a(
; CHECK-NEXT:    [[NE:%.*]] = icmp ne i64 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[NE]])
; CHECK-NEXT:    [[RET:%.*]] = icmp ugt i64 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[RET]]
;
  %ne = icmp ne i64 %a, %b
  tail call void @llvm.assume(i1 %ne)
  %ret = icmp ugt i64 %a, %b
  ret i1 %ret
}

define i1 @assume_1b(i64 %a, i64 %b) {
; CHECK-LABEL: @assume_1b(
; CHECK-NEXT:    [[NE:%.*]] = icmp ne i64 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[NE]])
; CHECK-NEXT:    [[RET:%.*]] = icmp uge i64 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[RET]]
;
  %ne = icmp ne i64 %a, %b
  tail call void @llvm.assume(i1 %ne)
  %ret = icmp uge i64 %a, %b
  ret i1 %ret
}

define i1 @assume_2a(i64 %a, i64 %b) {
; CHECK-LABEL: @assume_2a(
; CHECK-NEXT:    [[NE:%.*]] = icmp ne i64 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[NE]])
; CHECK-NEXT:    [[RET:%.*]] = icmp ult i64 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[RET]]
;
  %ne = icmp ne i64 %a, %b
  tail call void @llvm.assume(i1 %ne)
  %ret = icmp ult i64 %a, %b
  ret i1 %ret
}

define i1 @assume_2b(i64 %a, i64 %b) {
; CHECK-LABEL: @assume_2b(
; CHECK-NEXT:    [[NE:%.*]] = icmp ne i64 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[NE]])
; CHECK-NEXT:    [[RET:%.*]] = icmp ule i64 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[RET]]
;
  %ne = icmp ne i64 %a, %b
  tail call void @llvm.assume(i1 %ne)
  %ret = icmp ule i64 %a, %b
  ret i1 %ret
}

; TODO: extend to support signed comparisons
define i1 @assume_3a(i64 %a, i64 %b) {
; CHECK-LABEL: @assume_3a(
; CHECK-NEXT:    [[NE:%.*]] = icmp ne i64 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[NE]])
; CHECK-NEXT:    [[RET:%.*]] = icmp sgt i64 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[RET]]
;
  %ne = icmp ne i64 %a, %b
  tail call void @llvm.assume(i1 %ne)
  %ret = icmp sgt i64 %a, %b
  ret i1 %ret
}

define i1 @assume_3b(i64 %a, i64 %b) {
; CHECK-LABEL: @assume_3b(
; CHECK-NEXT:    [[NE:%.*]] = icmp ne i64 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[NE]])
; CHECK-NEXT:    [[RET:%.*]] = icmp sge i64 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[RET]]
;
  %ne = icmp ne i64 %a, %b
  tail call void @llvm.assume(i1 %ne)
  %ret = icmp sge i64 %a, %b
  ret i1 %ret
}

define i1 @assume_4a(i64 %a, i64 %b) {
; CHECK-LABEL: @assume_4a(
; CHECK-NEXT:    [[NE:%.*]] = icmp ne i64 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[NE]])
; CHECK-NEXT:    [[RET:%.*]] = icmp slt i64 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[RET]]
;
  %ne = icmp ne i64 %a, %b
  tail call void @llvm.assume(i1 %ne)
  %ret = icmp slt i64 %a, %b
  ret i1 %ret
}

define i1 @assume_4b(i64 %a, i64 %b) {
; CHECK-LABEL: @assume_4b(
; CHECK-NEXT:    [[NE:%.*]] = icmp ne i64 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[NE]])
; CHECK-NEXT:    [[RET:%.*]] = icmp sle i64 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[RET]]
;
  %ne = icmp ne i64 %a, %b
  tail call void @llvm.assume(i1 %ne)
  %ret = icmp sle i64 %a, %b
  ret i1 %ret
}
