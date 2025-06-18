! RUN: bbc -hlfir=false -fwrapv -o - --outline-intrinsics %s | FileCheck %s

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "p"} {
! CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 8 : i64
! CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 2 : i64
! CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 7 : i64
! CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 0 : i64
! CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 3 : i64
! CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 2 : index
! CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 4 : index
! CHECK-DAG:           %[[VAL_7:.*]] = arith.constant {{[0-9]+}} : i32
! CHECK-DAG:           %[[VAL_8:.*]] = arith.constant {{[0-9]+}} : i32
! CHECK-DAG:           %[[VAL_9:.*]] = arith.constant {{[0-9]+}} : i32
! CHECK-DAG:           %[[VAL_10:.*]] = arith.constant {{[0-9]+}} : i32
! CHECK-DAG:           %[[VAL_11:.*]] = arith.constant {{[0-9]+}} : i32
! CHECK-DAG:           %[[VAL_12:.*]] = arith.constant {{[0-9]+}} : i32
! CHECK-DAG:           %[[VAL_13:.*]] = arith.constant {{[0-9]+}} : i32
! CHECK-DAG:           %[[VAL_14:.*]] = arith.constant {{[0-9]+}} : i32
! CHECK-DAG:           %[[VAL_15:.*]] = arith.constant {{[0-9]+}} : i32
! CHECK-DAG:           %[[VAL_16:.*]] = arith.constant 6 : i32
! CHECK-DAG:           %[[VAL_17:.*]] = arith.constant 0 : index
! CHECK-DAG:           %[[VAL_18:.*]] = arith.constant 4 : i64
! CHECK-DAG:           %[[VAL_19:.*]] = arith.constant 1 : i64
! CHECK-DAG:           %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK-DAG:           %[[VAL_21:.*]] = arith.constant 1 : i32
! CHECK-DAG:           %[[VAL_22:.*]] = arith.constant 0 : i32
! CHECK-DAG:           %[[VAL_23:.*]] = arith.constant 3 : index
! CHECK-DAG:           %[[VAL_24:.*]] = arith.constant 10 : index
! CHECK-DAG:           %[[VAL_25:.*]] = fir.address_of(@_QFEa1) : !fir.ref<!fir.array<10x10xf32>>
! CHECK-DAG:           %[[VAL_26:.*]] = fir.alloca !fir.array<3xf32> {bindc_name = "a2", uniq_name = "_QFEa2"}
! CHECK-DAG:           %[[VAL_27:.*]] = fir.address_of(@_QFEa3) : !fir.ref<!fir.array<10xf32>>
! CHECK-DAG:           %[[VAL_28:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK-DAG:           %[[VAL_29:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = "iv", uniq_name = "_QFEiv"}
! CHECK-DAG:           %[[VAL_30:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFEj"}
! CHECK-DAG:           %[[VAL_31:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFEk"}
! CHECK:           fir.store %[[VAL_22]] to %[[VAL_31]] : !fir.ref<i32>
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_20]] : (index) -> i32
! CHECK:           cf.br ^bb1(%[[VAL_32]], %[[VAL_24]] : i32, index)
! CHECK:         ^bb1(%[[VAL_33:.*]]: i32, %[[VAL_34:.*]]: index):
! CHECK:           %[[VAL_35:.*]] = arith.cmpi sgt, %[[VAL_34]], %[[VAL_17]] : index
! CHECK:           cf.cond_br %[[VAL_35]], ^bb2, ^bb6
! CHECK:         ^bb2:
! CHECK:           fir.store %[[VAL_33]] to %[[VAL_30]] : !fir.ref<i32>
! CHECK:           cf.br ^bb3(%[[VAL_32]], %[[VAL_24]] : i32, index)
! CHECK:         ^bb3(%[[VAL_36:.*]]: i32, %[[VAL_37:.*]]: index):
! CHECK:           %[[VAL_38:.*]] = arith.cmpi sgt, %[[VAL_37]], %[[VAL_17]] : index
! CHECK:           cf.cond_br %[[VAL_38]], ^bb4, ^bb5
! CHECK:         ^bb4:
! CHECK:           fir.store %[[VAL_36]] to %[[VAL_28]] : !fir.ref<i32>
! CHECK:           %[[VAL_39:.*]] = fir.load %[[VAL_31]] : !fir.ref<i32>
! CHECK:           %[[VAL_40:.*]] = arith.addi %[[VAL_39]], %[[VAL_21]] : i32
! CHECK:           fir.store %[[VAL_40]] to %[[VAL_31]] : !fir.ref<i32>
! CHECK:           %[[VAL_41:.*]] = fir.load %[[VAL_31]] : !fir.ref<i32>
! CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i32) -> f32
! CHECK:           %[[VAL_43:.*]] = fir.call @fir.cos.contract.f32.f32(%[[VAL_42]]) fastmath<contract> : (f32) -> f32
! CHECK:           %[[VAL_44:.*]] = fir.load %[[VAL_28]] : !fir.ref<i32>
! CHECK:           %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i32) -> i64
! CHECK:           %[[VAL_46:.*]] = arith.subi %[[VAL_45]], %[[VAL_19]] : i64
! CHECK:           %[[VAL_47:.*]] = fir.load %[[VAL_30]] : !fir.ref<i32>
! CHECK:           %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i32) -> i64
! CHECK:           %[[VAL_49:.*]] = arith.subi %[[VAL_48]], %[[VAL_19]] : i64
! CHECK:           %[[VAL_50:.*]] = fir.coordinate_of %[[VAL_25]], %[[VAL_46]], %[[VAL_49]] : (!fir.ref<!fir.array<10x10xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:           fir.store %[[VAL_43]] to %[[VAL_50]] : !fir.ref<f32>
! CHECK:           %[[VAL_51:.*]] = fir.load %[[VAL_28]] : !fir.ref<i32>
! CHECK:           %[[VAL_52:.*]] = arith.addi %[[VAL_51]], %[[VAL_32]] : i32
! CHECK:           %[[VAL_53:.*]] = arith.subi %[[VAL_37]], %[[VAL_20]] : index
! CHECK:           cf.br ^bb3(%[[VAL_52]], %[[VAL_53]] : i32, index)
! CHECK:         ^bb5:
! CHECK:           fir.store %[[VAL_36]] to %[[VAL_28]] : !fir.ref<i32>
! CHECK:           %[[VAL_54:.*]] = fir.load %[[VAL_31]] : !fir.ref<i32>
! CHECK:           %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (i32) -> f32
! CHECK:           %[[VAL_56:.*]] = fir.call @fir.sin.contract.f32.f32(%[[VAL_55]]) fastmath<contract> : (f32) -> f32
! CHECK:           %[[VAL_57:.*]] = fir.load %[[VAL_30]] : !fir.ref<i32>
! CHECK:           %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i32) -> i64
! CHECK:           %[[VAL_59:.*]] = arith.subi %[[VAL_58]], %[[VAL_19]] : i64
! CHECK:           %[[VAL_60:.*]] = fir.coordinate_of %[[VAL_27]], %[[VAL_59]] : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
! CHECK:           fir.store %[[VAL_56]] to %[[VAL_60]] : !fir.ref<f32>
! CHECK:           %[[VAL_61:.*]] = fir.load %[[VAL_30]] : !fir.ref<i32>
! CHECK:           %[[VAL_62:.*]] = arith.addi %[[VAL_61]], %[[VAL_32]] : i32
! CHECK:           %[[VAL_63:.*]] = arith.subi %[[VAL_34]], %[[VAL_20]] : index
! CHECK:           cf.br ^bb1(%[[VAL_62]], %[[VAL_63]] : i32, index)
! CHECK:         ^bb6:
! CHECK:           fir.store %[[VAL_33]] to %[[VAL_30]] : !fir.ref<i32>
! CHECK:           %[[VAL_64:.*]] = fir.shape %[[VAL_23]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_65:.*]] = fir.undefined index
! CHECK:           %[[VAL_66:.*]] = fir.shape %[[VAL_24]], %[[VAL_24]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_67:.*]] = fir.slice %[[VAL_18]], %[[VAL_65]], %[[VAL_65]], %[[VAL_5]], %[[VAL_24]], %[[VAL_23]] : (i64, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           cf.br ^bb7(%[[VAL_17]], %[[VAL_23]] : index, index)
! CHECK:         ^bb7(%[[VAL_68:.*]]: index, %[[VAL_69:.*]]: index):
! CHECK:           %[[VAL_70:.*]] = arith.cmpi sgt, %[[VAL_69]], %[[VAL_17]] : index
! CHECK:           cf.cond_br %[[VAL_70]], ^bb8, ^bb9
! CHECK:         ^bb8:
! CHECK:           %[[VAL_71:.*]] = arith.addi %[[VAL_68]], %[[VAL_20]] : index
! CHECK:           %[[VAL_72:.*]] = fir.array_coor %[[VAL_25]](%[[VAL_66]]) {{\[}}%[[VAL_67]]] %[[VAL_6]], %[[VAL_71]] : (!fir.ref<!fir.array<10x10xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_73:.*]] = fir.load %[[VAL_72]] : !fir.ref<f32>
! CHECK:           %[[VAL_74:.*]] = fir.array_coor %[[VAL_26]](%[[VAL_64]]) %[[VAL_71]] : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[VAL_73]] to %[[VAL_74]] : !fir.ref<f32>
! CHECK:           %[[VAL_75:.*]] = arith.subi %[[VAL_69]], %[[VAL_20]] : index
! CHECK:           cf.br ^bb7(%[[VAL_71]], %[[VAL_75]] : index, index)
! CHECK:         ^bb9:
! CHECK:           %[[VAL_76:.*]] = fir.coordinate_of %[[VAL_25]], %[[VAL_4]], %[[VAL_19]] : (!fir.ref<!fir.array<10x10xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:           %[[VAL_77:.*]] = fir.load %[[VAL_76]] : !fir.ref<f32>
! CHECK:           %[[VAL_78:.*]] = fir.coordinate_of %[[VAL_26]], %[[VAL_3]] : (!fir.ref<!fir.array<3xf32>>, i64) -> !fir.ref<f32>
! CHECK:           %[[VAL_79:.*]] = fir.load %[[VAL_78]] : !fir.ref<f32>
! CHECK:           %[[VAL_80:.*]] = arith.cmpf une, %[[VAL_77]], %[[VAL_79]] fastmath<contract> : f32
! CHECK:           cf.cond_br %[[VAL_80]], ^bb10, ^bb11
! CHECK:         ^bb10:
! CHECK:           %[[VAL_81:.*]] = fir.address_of(@_QQclX3bee04f64cc15c75e483e8463401fb13) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_82:.*]] = fir.convert %[[VAL_81]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_83:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_16]], %[[VAL_82]], %[[VAL_15]]) {{.+}} (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_84:.*]] = fir.address_of(@_QQclX6D69736D617463682031) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_85:.*]] = fir.convert %[[VAL_84]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_86:.*]] = fir.convert %[[VAL_24]] : (index) -> i64
! CHECK:           %[[VAL_87:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_83]], %[[VAL_85]], %[[VAL_86]]) {{.+}} (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:           %[[VAL_88:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_83]], %[[VAL_79]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_89:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_83]], %[[VAL_77]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_90:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_83]]) {{.+}} (!fir.ref<i8>) -> i32
! CHECK:           cf.br ^bb11
! CHECK:         ^bb11:
! CHECK:           %[[VAL_91:.*]] = fir.coordinate_of %[[VAL_25]], %[[VAL_4]], %[[VAL_18]] : (!fir.ref<!fir.array<10x10xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:           %[[VAL_92:.*]] = fir.load %[[VAL_91]] : !fir.ref<f32>
! CHECK:           %[[VAL_93:.*]] = fir.coordinate_of %[[VAL_26]], %[[VAL_19]] : (!fir.ref<!fir.array<3xf32>>, i64) -> !fir.ref<f32>
! CHECK:           %[[VAL_94:.*]] = fir.load %[[VAL_93]] : !fir.ref<f32>
! CHECK:           %[[VAL_95:.*]] = arith.cmpf une, %[[VAL_92]], %[[VAL_94]] fastmath<contract> : f32
! CHECK:           cf.cond_br %[[VAL_95]], ^bb12, ^bb13
! CHECK:         ^bb12:
! CHECK:           %[[VAL_96:.*]] = fir.address_of(@_QQclX3bee04f64cc15c75e483e8463401fb13) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_97:.*]] = fir.convert %[[VAL_96]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_98:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_16]], %[[VAL_97]], %[[VAL_14]]) {{.+}} (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_99:.*]] = fir.address_of(@_QQclX6D69736D617463682032) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_100:.*]] = fir.convert %[[VAL_99]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_101:.*]] = fir.convert %[[VAL_24]] : (index) -> i64
! CHECK:           %[[VAL_102:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_98]], %[[VAL_100]], %[[VAL_101]]) {{.+}} (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:           %[[VAL_103:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_98]], %[[VAL_94]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_104:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_98]], %[[VAL_92]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_105:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_98]]) {{.+}} (!fir.ref<i8>) -> i32
! CHECK:           cf.br ^bb13
! CHECK:         ^bb13:
! CHECK:           %[[VAL_106:.*]] = fir.coordinate_of %[[VAL_25]], %[[VAL_4]], %[[VAL_2]] : (!fir.ref<!fir.array<10x10xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:           %[[VAL_107:.*]] = fir.load %[[VAL_106]] : !fir.ref<f32>
! CHECK:           %[[VAL_108:.*]] = fir.coordinate_of %[[VAL_26]], %[[VAL_1]] : (!fir.ref<!fir.array<3xf32>>, i64) -> !fir.ref<f32>
! CHECK:           %[[VAL_109:.*]] = fir.load %[[VAL_108]] : !fir.ref<f32>
! CHECK:           %[[VAL_110:.*]] = arith.cmpf une, %[[VAL_107]], %[[VAL_109]] fastmath<contract> : f32
! CHECK:           cf.cond_br %[[VAL_110]], ^bb14, ^bb15
! CHECK:         ^bb14:
! CHECK:           %[[VAL_111:.*]] = fir.address_of(@_QQclX3bee04f64cc15c75e483e8463401fb13) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_112:.*]] = fir.convert %[[VAL_111]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_113:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_16]], %[[VAL_112]], %[[VAL_13]]) {{.+}} (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_114:.*]] = fir.address_of(@_QQclX6D69736D617463682033) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_115:.*]] = fir.convert %[[VAL_114]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_116:.*]] = fir.convert %[[VAL_24]] : (index) -> i64
! CHECK:           %[[VAL_117:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_113]], %[[VAL_115]], %[[VAL_116]]) {{.+}} (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:           %[[VAL_118:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_113]], %[[VAL_109]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_119:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_113]], %[[VAL_107]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_120:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_113]]) {{.+}} (!fir.ref<i8>) -> i32
! CHECK:           cf.br ^bb15
! CHECK:         ^bb15:
! CHECK:           %[[VAL_121:.*]] = fir.shape %[[VAL_24]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_122:.*]] = fir.slice %[[VAL_20]], %[[VAL_24]], %[[VAL_6]] : (index, index, index) -> !fir.slice<1>
! CHECK:           cf.br ^bb16(%[[VAL_17]], %[[VAL_23]] : index, index)
! CHECK:         ^bb16(%[[VAL_123:.*]]: index, %[[VAL_124:.*]]: index):
! CHECK:           %[[VAL_125:.*]] = arith.cmpi sgt, %[[VAL_124]], %[[VAL_17]] : index
! CHECK:           cf.cond_br %[[VAL_125]], ^bb17, ^bb18
! CHECK:         ^bb17:
! CHECK:           %[[VAL_126:.*]] = arith.addi %[[VAL_123]], %[[VAL_20]] : index
! CHECK:           %[[VAL_127:.*]] = fir.array_coor %[[VAL_26]](%[[VAL_64]]) %[[VAL_126]] : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_128:.*]] = fir.load %[[VAL_127]] : !fir.ref<f32>
! CHECK:           %[[VAL_129:.*]] = fir.array_coor %[[VAL_27]](%[[VAL_121]]) {{\[}}%[[VAL_122]]] %[[VAL_126]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[VAL_128]] to %[[VAL_129]] : !fir.ref<f32>
! CHECK:           %[[VAL_130:.*]] = arith.subi %[[VAL_124]], %[[VAL_20]] : index
! CHECK:           cf.br ^bb16(%[[VAL_126]], %[[VAL_130]] : index, index)
! CHECK:         ^bb18:
! CHECK:           %[[VAL_131:.*]] = fir.coordinate_of %[[VAL_27]], %[[VAL_3]] : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
! CHECK:           %[[VAL_132:.*]] = fir.load %[[VAL_131]] : !fir.ref<f32>
! CHECK:           %[[VAL_133:.*]] = arith.cmpf une, %[[VAL_77]], %[[VAL_132]] fastmath<contract> : f32
! CHECK:           cf.cond_br %[[VAL_133]], ^bb19, ^bb20
! CHECK:         ^bb19:
! CHECK:           %[[VAL_134:.*]] = fir.address_of(@_QQclX3bee04f64cc15c75e483e8463401fb13) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_135:.*]] = fir.convert %[[VAL_134]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_136:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_16]], %[[VAL_135]], %[[VAL_12]]) {{.+}} (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_137:.*]] = fir.address_of(@_QQclX6D69736D617463682034) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_138:.*]] = fir.convert %[[VAL_137]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_139:.*]] = fir.convert %[[VAL_24]] : (index) -> i64
! CHECK:           %[[VAL_140:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_136]], %[[VAL_138]], %[[VAL_139]]) {{.+}} (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:           %[[VAL_141:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_136]], %[[VAL_77]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_142:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_136]], %[[VAL_132]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_143:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_136]]) {{.+}} (!fir.ref<i8>) -> i32
! CHECK:           cf.br ^bb20
! CHECK:         ^bb20:
! CHECK:           %[[VAL_144:.*]] = fir.coordinate_of %[[VAL_27]], %[[VAL_18]] : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
! CHECK:           %[[VAL_145:.*]] = fir.load %[[VAL_144]] : !fir.ref<f32>
! CHECK:           %[[VAL_146:.*]] = arith.cmpf une, %[[VAL_92]], %[[VAL_145]] fastmath<contract> : f32
! CHECK:           cf.cond_br %[[VAL_146]], ^bb21, ^bb22
! CHECK:         ^bb21:
! CHECK:           %[[VAL_147:.*]] = fir.address_of(@_QQclX3bee04f64cc15c75e483e8463401fb13) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_148:.*]] = fir.convert %[[VAL_147]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_149:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_16]], %[[VAL_148]], %[[VAL_11]]) {{.+}} (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_150:.*]] = fir.address_of(@_QQclX6D69736D617463682035) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_151:.*]] = fir.convert %[[VAL_150]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_152:.*]] = fir.convert %[[VAL_24]] : (index) -> i64
! CHECK:           %[[VAL_153:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_149]], %[[VAL_151]], %[[VAL_152]]) {{.+}} (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:           %[[VAL_154:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_149]], %[[VAL_92]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_155:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_149]], %[[VAL_145]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_156:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_149]]) {{.+}} (!fir.ref<i8>) -> i32
! CHECK:           cf.br ^bb22
! CHECK:         ^bb22:
! CHECK:           %[[VAL_157:.*]] = fir.coordinate_of %[[VAL_27]], %[[VAL_0]] : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
! CHECK:           %[[VAL_158:.*]] = fir.load %[[VAL_157]] : !fir.ref<f32>
! CHECK:           %[[VAL_159:.*]] = arith.cmpf une, %[[VAL_107]], %[[VAL_158]] fastmath<contract> : f32
! CHECK:           cf.cond_br %[[VAL_159]], ^bb23, ^bb24
! CHECK:         ^bb23:
! CHECK:           %[[VAL_160:.*]] = fir.address_of(@_QQclX3bee04f64cc15c75e483e8463401fb13) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_161:.*]] = fir.convert %[[VAL_160]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_162:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_16]], %[[VAL_161]], %[[VAL_10]]) {{.+}} (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_163:.*]] = fir.address_of(@_QQclX6D69736D617463682036) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_164:.*]] = fir.convert %[[VAL_163]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_165:.*]] = fir.convert %[[VAL_24]] : (index) -> i64
! CHECK:           %[[VAL_166:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_162]], %[[VAL_164]], %[[VAL_165]]) {{.+}} (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:           %[[VAL_167:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_162]], %[[VAL_107]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_168:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_162]], %[[VAL_158]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_169:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_162]]) {{.+}} (!fir.ref<i8>) -> i32
! CHECK:           cf.br ^bb24
! CHECK:         ^bb24:
! CHECK:           %[[VAL_170:.*]] = fir.address_of(@_QQro.3xi4.0) : !fir.ref<!fir.array<3xi32>>
! CHECK:           cf.br ^bb25(%[[VAL_17]], %[[VAL_23]] : index, index)
! CHECK:         ^bb25(%[[VAL_171:.*]]: index, %[[VAL_172:.*]]: index):
! CHECK:           %[[VAL_173:.*]] = arith.cmpi sgt, %[[VAL_172]], %[[VAL_17]] : index
! CHECK:           cf.cond_br %[[VAL_173]], ^bb26, ^bb27
! CHECK:         ^bb26:
! CHECK:           %[[VAL_174:.*]] = arith.addi %[[VAL_171]], %[[VAL_20]] : index
! CHECK:           %[[VAL_175:.*]] = fir.array_coor %[[VAL_170]](%[[VAL_64]]) %[[VAL_174]] : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_176:.*]] = fir.load %[[VAL_175]] : !fir.ref<i32>
! CHECK:           %[[VAL_177:.*]] = fir.array_coor %[[VAL_29]](%[[VAL_64]]) %[[VAL_174]] : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:           fir.store %[[VAL_176]] to %[[VAL_177]] : !fir.ref<i32>
! CHECK:           %[[VAL_178:.*]] = arith.subi %[[VAL_172]], %[[VAL_20]] : index
! CHECK:           cf.br ^bb25(%[[VAL_174]], %[[VAL_178]] : index, index)
! CHECK:         ^bb27:
! CHECK:           %[[VAL_179:.*]] = fir.allocmem !fir.array<3xf32>
! CHECK:           cf.br ^bb28(%[[VAL_17]], %[[VAL_23]] : index, index)
! CHECK:         ^bb28(%[[VAL_180:.*]]: index, %[[VAL_181:.*]]: index):
! CHECK:           %[[VAL_182:.*]] = arith.cmpi sgt, %[[VAL_181]], %[[VAL_17]] : index
! CHECK:           cf.cond_br %[[VAL_182]], ^bb29, ^bb30(%[[VAL_17]], %[[VAL_23]] : index, index)
! CHECK:         ^bb29:
! CHECK:           %[[VAL_183:.*]] = arith.addi %[[VAL_180]], %[[VAL_20]] : index
! CHECK:           %[[VAL_184:.*]] = fir.array_coor %[[VAL_26]](%[[VAL_64]]) %[[VAL_183]] : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_185:.*]] = fir.array_coor %[[VAL_179]](%[[VAL_64]]) %[[VAL_183]] : (!fir.heap<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_186:.*]] = fir.load %[[VAL_184]] : !fir.ref<f32>
! CHECK:           fir.store %[[VAL_186]] to %[[VAL_185]] : !fir.ref<f32>
! CHECK:           %[[VAL_187:.*]] = arith.subi %[[VAL_181]], %[[VAL_20]] : index
! CHECK:           cf.br ^bb28(%[[VAL_183]], %[[VAL_187]] : index, index)
! CHECK:         ^bb30(%[[VAL_188:.*]]: index, %[[VAL_189:.*]]: index):
! CHECK:           %[[VAL_190:.*]] = arith.cmpi sgt, %[[VAL_189]], %[[VAL_17]] : index
! CHECK:           cf.cond_br %[[VAL_190]], ^bb31, ^bb32(%[[VAL_17]], %[[VAL_23]] : index, index)
! CHECK:         ^bb31:
! CHECK:           %[[VAL_191:.*]] = arith.addi %[[VAL_188]], %[[VAL_20]] : index
! CHECK:           %[[VAL_192:.*]] = fir.array_coor %[[VAL_29]](%[[VAL_64]]) %[[VAL_191]] : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_193:.*]] = fir.load %[[VAL_192]] : !fir.ref<i32>
! CHECK:           %[[VAL_194:.*]] = fir.convert %[[VAL_193]] : (i32) -> index
! CHECK:           %[[VAL_195:.*]] = fir.array_coor %[[VAL_26]](%[[VAL_64]]) %[[VAL_194]] : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_196:.*]] = fir.load %[[VAL_195]] : !fir.ref<f32>
! CHECK:           %[[VAL_197:.*]] = fir.array_coor %[[VAL_179]](%[[VAL_64]]) %[[VAL_191]] : (!fir.heap<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[VAL_196]] to %[[VAL_197]] : !fir.ref<f32>
! CHECK:           %[[VAL_198:.*]] = arith.subi %[[VAL_189]], %[[VAL_20]] : index
! CHECK:           cf.br ^bb30(%[[VAL_191]], %[[VAL_198]] : index, index)
! CHECK:         ^bb32(%[[VAL_199:.*]]: index, %[[VAL_200:.*]]: index):
! CHECK:           %[[VAL_201:.*]] = arith.cmpi sgt, %[[VAL_200]], %[[VAL_17]] : index
! CHECK:           cf.cond_br %[[VAL_201]], ^bb33, ^bb34
! CHECK:         ^bb33:
! CHECK:           %[[VAL_202:.*]] = arith.addi %[[VAL_199]], %[[VAL_20]] : index
! CHECK:           %[[VAL_203:.*]] = fir.array_coor %[[VAL_179]](%[[VAL_64]]) %[[VAL_202]] : (!fir.heap<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_204:.*]] = fir.array_coor %[[VAL_26]](%[[VAL_64]]) %[[VAL_202]] : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_205:.*]] = fir.load %[[VAL_203]] : !fir.ref<f32>
! CHECK:           fir.store %[[VAL_205]] to %[[VAL_204]] : !fir.ref<f32>
! CHECK:           %[[VAL_206:.*]] = arith.subi %[[VAL_200]], %[[VAL_20]] : index
! CHECK:           cf.br ^bb32(%[[VAL_202]], %[[VAL_206]] : index, index)
! CHECK:         ^bb34:
! CHECK:           fir.freemem %[[VAL_179]] : !fir.heap<!fir.array<3xf32>>
! CHECK:           %[[VAL_207:.*]] = arith.cmpf une, %[[VAL_77]], %[[VAL_94]] fastmath<contract> : f32
! CHECK:           cf.cond_br %[[VAL_207]], ^bb35, ^bb36
! CHECK:         ^bb35:
! CHECK:           %[[VAL_208:.*]] = fir.address_of(@_QQclX3bee04f64cc15c75e483e8463401fb13) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_209:.*]] = fir.convert %[[VAL_208]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_210:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_16]], %[[VAL_209]], %[[VAL_9]]) {{.+}} (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_211:.*]] = fir.address_of(@_QQclX6D69736D617463682037) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_212:.*]] = fir.convert %[[VAL_211]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_213:.*]] = fir.convert %[[VAL_24]] : (index) -> i64
! CHECK:           %[[VAL_214:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_210]], %[[VAL_212]], %[[VAL_213]]) {{.+}} (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:           %[[VAL_215:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_210]], %[[VAL_77]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_216:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_210]], %[[VAL_94]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_217:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_210]]) {{.+}} (!fir.ref<i8>) -> i32
! CHECK:           cf.br ^bb36
! CHECK:         ^bb36:
! CHECK:           %[[VAL_218:.*]] = arith.cmpf une, %[[VAL_92]], %[[VAL_109]] fastmath<contract> : f32
! CHECK:           cf.cond_br %[[VAL_218]], ^bb37, ^bb38
! CHECK:         ^bb37:
! CHECK:           %[[VAL_219:.*]] = fir.address_of(@_QQclX3bee04f64cc15c75e483e8463401fb13) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_220:.*]] = fir.convert %[[VAL_219]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_221:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_16]], %[[VAL_220]], %[[VAL_8]]) {{.+}} (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_222:.*]] = fir.address_of(@_QQclX6D69736D617463682038) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_223:.*]] = fir.convert %[[VAL_222]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_224:.*]] = fir.convert %[[VAL_24]] : (index) -> i64
! CHECK:           %[[VAL_225:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_221]], %[[VAL_223]], %[[VAL_224]]) {{.+}} (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:           %[[VAL_226:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_221]], %[[VAL_92]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_227:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_221]], %[[VAL_109]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_228:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_221]]) {{.+}} (!fir.ref<i8>) -> i32
! CHECK:           cf.br ^bb38
! CHECK:         ^bb38:
! CHECK:           %[[VAL_229:.*]] = arith.cmpf une, %[[VAL_107]], %[[VAL_79]] fastmath<contract> : f32
! CHECK:           cf.cond_br %[[VAL_229]], ^bb39, ^bb40
! CHECK:         ^bb39:
! CHECK:           %[[VAL_230:.*]] = fir.address_of(@_QQclX3bee04f64cc15c75e483e8463401fb13) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_231:.*]] = fir.convert %[[VAL_230]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_232:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_16]], %[[VAL_231]], %[[VAL_7]]) {{.+}} (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_233:.*]] = fir.address_of(@_QQclX6D69736D617463682039) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[VAL_234:.*]] = fir.convert %[[VAL_233]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_235:.*]] = fir.convert %[[VAL_24]] : (index) -> i64
! CHECK:           %[[VAL_236:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_232]], %[[VAL_234]], %[[VAL_235]]) {{.+}} (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:           %[[VAL_237:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_232]], %[[VAL_107]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_238:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_232]], %[[VAL_79]]) {{.+}} (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_239:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_232]]) {{.+}} (!fir.ref<i8>) -> i32
! CHECK:           cf.br ^bb40
! CHECK:         ^bb40:
! CHECK:           return
! CHECK:         }


program p
  real :: a1(10,10)
  real :: a2(3)
  real :: a3(10)
  integer iv(3)
  integer k

  k = 0
  do j = 1, 10
     do i = 1, 10
        k = k + 1
        a1(i,j) = cos(real(k))
     end do
     a3(j) = sin(real(k))
  end do

  a2 = a1(4, 2:10:3)

  if (a1(4,2) .ne. a2(1)) print *, "mismatch 1", a2(1), a1(4,2)
  if (a1(4,5) .ne. a2(2)) print *, "mismatch 2", a2(2), a1(4,5)
  if (a1(4,8) .ne. a2(3)) print *, "mismatch 3", a2(3), a1(4,8)

  a3(1:10:4) = a2

  if (a1(4,2) .ne. a3(1)) print *, "mismatch 4", a1(4,2), a3(1)
  if (a1(4,5) .ne. a3(5)) print *, "mismatch 5", a1(4,5), a3(5)
  if (a1(4,8) .ne. a3(9)) print *, "mismatch 6", a1(4,8), a3(9)

  iv = (/ 3, 1, 2 /)

  a2 = a2(iv)

  if (a1(4,2) .ne. a2(2)) print *, "mismatch 7", a1(4,2), a2(2)
  if (a1(4,5) .ne. a2(3)) print *, "mismatch 8", a1(4,5), a2(3)
  if (a1(4,8) .ne. a2(1)) print *, "mismatch 9", a1(4,8), a2(1)

end program p

! CHECK-LABEL: func @_QPsub(
! CHECK-SAME:               %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}) {
! CHECK-DAG:     %[[VAL_1:.*]] = arith.constant 5 : index
! CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 4 : index
! CHECK-DAG:     %[[VAL_6:.*]] = arith.constant 6 : i32
! CHECK-DAG:     %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_8:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
! CHECK:         %[[VAL_10:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_12:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_6]], %[[VAL_11]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_13:.*]] = fir.address_of(@_QQclX61203D20) : !fir.ref<!fir.char<1,4>>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
! CHECK:         %[[VAL_16:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_12]], %[[VAL_14]], %[[VAL_15]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:         %[[VAL_17:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_18:.*]] = fir.slice %[[VAL_3]], %[[VAL_1]], %[[VAL_2]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_19:.*]] = fir.embox %[[VAL_9]](%[[VAL_17]]) {{\[}}%[[VAL_18]]] : (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<3x!fir.char<1>>>
! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (!fir.box<!fir.array<3x!fir.char<1>>>) -> !fir.box<none>
! CHECK:         %[[VAL_21:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_12]], %[[VAL_20]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:         %[[VAL_22:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_12]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:         return
! CHECK:       }

! Slice operation on array of CHARACTER
subroutine sub(a)
  character :: a(10)
  print *, "a = ", a(1:5:2)
end subroutine sub
