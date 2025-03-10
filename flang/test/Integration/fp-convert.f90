program float_to_uint_conversion
  use, intrinsic :: iso_c_binding
  use, intrinsic :: ieee_arithmetic
  use iso_fortran_env, only : uint8, uint16, uint32, uint64

  implicit none
  real(kind=16) :: fhuge, nan, inf, ninf, ftiny
  integer(c_int8_t) :: i8
  integer(c_int16_t) :: i16
  integer(c_int32_t) :: i32
  integer(c_int64_t) :: i64
  unsigned(kind=uint8) :: u8
  unsigned(kind=uint16) :: u16
  unsigned(kind=uint32) :: u32
  unsigned(kind=uint64) :: u64

  fhuge = huge(fhuge)
  nan = ieee_value(fhuge, ieee_quiet_nan)
  inf = ieee_value(fhuge, ieee_positive_inf)
  ninf = ieee_value(fhuge, ieee_negative_inf)

  i8 = int(fhuge, c_int8_t)
  i16 = int(fhuge, c_int16_t)
  i32 = int(fhuge, c_int32_t)
  i64 = int(fhuge, c_int64_t)

  u8 = uint(fhuge, uint8)
  u16 = uint(fhuge, uint16)
  u32 = uint(fhuge, uint32)
  u64 = uint(fhuge, uint64)
  
  print *, "Original float value:", fhuge

  print *, "Converted to 8-bit integer:", i8
  print *, "Converted to 16-bit integer:", i16
  print *, "Converted to 32-bit integer:", i32
  print *, "Converted to 64-bit integer:", i64
  print *, "Converted to 8-bit unsigned:", u8
  print *, "Converted to 16-bit unsigned:", u16
  print *, "Converted to 32-bit unsigned:", u32
  print *, "Converted to 64-bit unsigned:", u64

  ! CHECK:  Converted to 8-bit integer: 127
  ! CHECK:  Converted to 16-bit integer: 32767
  ! CHECK:  Converted to 32-bit integer: 2147483647
  ! CHECK:  Converted to 64-bit integer: 9223372036854775807
  ! CHECK:  Converted to 8-bit unsigned: 127
  ! CHECK:  Converted to 16-bit unsigned: 32767
  ! CHECK:  Converted to 32-bit unsigned: 2147483647
  ! CHECK:  Converted to 64-bit unsigned: 9223372036854775807
  
  ! CHECK: llvm.call_intrinsic "llvm.fptosi.sat"(%{{[0-9]+}}) : (f128) -> i8
  ! CHECK: llvm.call_intrinsic "llvm.fptosi.sat"(%{{[0-9]+}}) : (f128) -> i16
  ! CHECK: llvm.call_intrinsic "llvm.fptosi.sat"(%{{[0-9]+}}) : (f128) -> i32
  ! CHECK: llvm.call_intrinsic "llvm.fptosi.sat"(%{{[0-9]+}}) : (f128) -> i64
  ! CHECK: llvm.call_intrinsic "llvm.fptosi.sat"(%{{[0-9]+}}) : (f128) -> i8
  ! CHECK: llvm.call_intrinsic "llvm.fptosi.sat"(%{{[0-9]+}}) : (f128) -> i16
  ! CHECK: llvm.call_intrinsic "llvm.fptosi.sat"(%{{[0-9]+}}) : (f128) -> i32
  ! CHECK: llvm.call_intrinsic "llvm.fptosi.sat"(%{{[0-9]+}}) : (f128) -> i64

  ftiny = tiny(ftiny)

  i8 = int(ftiny, c_int8_t)
  i16 = int(ftiny, c_int16_t)
  i32 = int(ftiny, c_int32_t)
  i64 = int(ftiny, c_int64_t)

  u8 = uint(ftiny, uint8)
  u16 = uint(ftiny, uint16)
  u32 = uint(ftiny, uint32)
  u64 = uint(ftiny, uint64)
  
  print *, "Original float value:", ftiny

  print *, "Converted to 8-bit integer:", i8
  print *, "Converted to 16-bit integer:", i16
  print *, "Converted to 32-bit integer:", i32
  print *, "Converted to 64-bit integer:", i64
  print *, "Converted to 8-bit unsigned:", u8
  print *, "Converted to 16-bit unsigned:", u16
  print *, "Converted to 32-bit unsigned:", u32
  print *, "Converted to 64-bit unsigned:", u64

  i8 = int(nan, c_int8_t)
  i16 = int(nan, c_int16_t)
  i32 = int(nan, c_int32_t)
  i64 = int(nan, c_int64_t)

  u8 = uint(nan, uint8)
  u16 = uint(nan, uint16)
  u32 = uint(nan, uint32)
  u64 = uint(nan, uint64)
  
  print *, "Original float value:", nan

  print *, "Converted to 8-bit integer:", i8
  print *, "Converted to 16-bit integer:", i16
  print *, "Converted to 32-bit integer:", i32
  print *, "Converted to 64-bit integer:", i64
  print *, "Converted to 8-bit unsigned:", u8
  print *, "Converted to 16-bit unsigned:", u16
  print *, "Converted to 32-bit unsigned:", u32
  print *, "Converted to 64-bit unsigned:", u64

  i8 = int(inf, c_int8_t)
  i16 = int(inf, c_int16_t)
  i32 = int(inf, c_int32_t)
  i64 = int(inf, c_int64_t)

  u8 = uint(inf, uint8)
  u16 = uint(inf, uint16)
  u32 = uint(inf, uint32)
  u64 = uint(inf, uint64)
  
  print *, "Original float value:", inf

  print *, "Converted to 8-bit integer:", i8
  print *, "Converted to 16-bit integer:", i16
  print *, "Converted to 32-bit integer:", i32
  print *, "Converted to 64-bit integer:", i64
  print *, "Converted to 8-bit unsigned:", u8
  print *, "Converted to 16-bit unsigned:", u16
  print *, "Converted to 32-bit unsigned:", u32
  print *, "Converted to 64-bit unsigned:", u64

  i8 = int(ninf, c_int8_t)
  i16 = int(ninf, c_int16_t)
  i32 = int(ninf, c_int32_t)
  i64 = int(ninf, c_int64_t)

  u8 = uint(ninf, uint8)
  u16 = uint(ninf, uint16)
  u32 = uint(ninf, uint32)
  u64 = uint(ninf, uint64)
  
  print *, "Original float value:", ninf

  print *, "Converted to 8-bit integer:", i8
  print *, "Converted to 16-bit integer:", i16
  print *, "Converted to 32-bit integer:", i32
  print *, "Converted to 64-bit integer:", i64
  print *, "Converted to 8-bit unsigned:", u8
  print *, "Converted to 16-bit unsigned:", u16
  print *, "Converted to 32-bit unsigned:", u32
  print *, "Converted to 64-bit unsigned:", u64
  
end program float_to_uint_conversion

