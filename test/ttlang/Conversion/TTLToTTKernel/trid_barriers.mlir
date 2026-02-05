// RUN: ttlang-opt --convert-ttl-to-ttkernel="use-trid-barriers=1" --canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=TTKERNEL
// RUN: ttlang-opt --convert-ttl-to-ttkernel="use-trid-barriers=1" --canonicalize --split-input-file %s | FileCheck %s --check-prefix=TTKERNEL_WRAP
// Summary: Regression tests for TRID-aware ttl.copy/ttl.wait lowering.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL-LABEL: func.func @trid_single_copy_wait_read
// TTKERNEL-DAG: %[[TRID:.*]] = arith.constant 0 : i32
// TTKERNEL-DAG: %[[NOC:.*]] = arith.constant 0 : i8
// TTKERNEL: ttkernel.noc_async_read_set_trid(%[[TRID]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL: ttkernel.noc_async_read_tile(
// TTKERNEL: ttkernel.noc_async_read_barrier_with_trid(%[[TRID]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL-NOT: ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: builtin.unrealized_conversion_cast
module {
  func.func @trid_single_copy_wait_read(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf = ttl.copy %slice, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL-LABEL: func.func @trid_two_copies_two_waits_read
// TTKERNEL-DAG: %[[TRID0:.*]] = arith.constant 0 : i32
// TTKERNEL-DAG: %[[TRID1:.*]] = arith.constant 1 : i32
// TTKERNEL-DAG: %[[NOC:.*]] = arith.constant 0 : i8
// TTKERNEL: ttkernel.noc_async_read_set_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL: ttkernel.noc_async_read_tile(
// TTKERNEL: ttkernel.noc_async_read_set_trid(%[[TRID1]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL: ttkernel.noc_async_read_tile(
// TTKERNEL: ttkernel.noc_async_read_barrier_with_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL: ttkernel.noc_async_read_barrier_with_trid(%[[TRID1]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL-NOT: ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: builtin.unrealized_conversion_cast
module {
  func.func @trid_two_copies_two_waits_read(%t0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, %t1: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %slice0 = ttl.tensor_slice %t0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %slice1 = ttl.tensor_slice %t1[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf0 = ttl.copy %slice0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf1 = ttl.copy %slice1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL_WRAP-LABEL: func.func @trid_wrap_reuse_read
// TTKERNEL_WRAP-DAG: %[[TRID0:.*]] = arith.constant 0 : i32
// TTKERNEL_WRAP-DAG: %[[NOC:.*]] = arith.constant 0 : i8
// TTKERNEL_WRAP: ttkernel.noc_async_read_set_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL_WRAP: ttkernel.noc_async_read_barrier_with_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL_WRAP: ttkernel.noc_async_read_set_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
module {
  func.func @trid_wrap_reuse_read(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb5 = ttl.bind_cb {cb_index = 5, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb6 = ttl.bind_cb {cb_index = 6, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb7 = ttl.bind_cb {cb_index = 7, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb8 = ttl.bind_cb {cb_index = 8, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb9 = ttl.bind_cb {cb_index = 9, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb10 = ttl.bind_cb {cb_index = 10, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb11 = ttl.bind_cb {cb_index = 11, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb12 = ttl.bind_cb {cb_index = 12, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb13 = ttl.bind_cb {cb_index = 13, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb14 = ttl.bind_cb {cb_index = 14, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb15 = ttl.bind_cb {cb_index = 15, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb16 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf0 = ttl.copy %slice, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf1 = ttl.copy %slice, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf2 = ttl.copy %slice, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf3 = ttl.copy %slice, %cb3 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf4 = ttl.copy %slice, %cb4 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf5 = ttl.copy %slice, %cb5 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf6 = ttl.copy %slice, %cb6 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf7 = ttl.copy %slice, %cb7 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf8 = ttl.copy %slice, %cb8 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf9 = ttl.copy %slice, %cb9 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf10 = ttl.copy %slice, %cb10 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf11 = ttl.copy %slice, %cb11 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf12 = ttl.copy %slice, %cb12 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf13 = ttl.copy %slice, %cb13 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf14 = ttl.copy %slice, %cb14 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf15 = ttl.copy %slice, %cb15 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf16 = ttl.copy %slice, %cb16 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    ttl.wait %xf2 : !ttl.transfer_handle<read>
    ttl.wait %xf3 : !ttl.transfer_handle<read>
    ttl.wait %xf4 : !ttl.transfer_handle<read>
    ttl.wait %xf5 : !ttl.transfer_handle<read>
    ttl.wait %xf6 : !ttl.transfer_handle<read>
    ttl.wait %xf7 : !ttl.transfer_handle<read>
    ttl.wait %xf8 : !ttl.transfer_handle<read>
    ttl.wait %xf9 : !ttl.transfer_handle<read>
    ttl.wait %xf10 : !ttl.transfer_handle<read>
    ttl.wait %xf11 : !ttl.transfer_handle<read>
    ttl.wait %xf12 : !ttl.transfer_handle<read>
    ttl.wait %xf13 : !ttl.transfer_handle<read>
    ttl.wait %xf14 : !ttl.transfer_handle<read>
    ttl.wait %xf15 : !ttl.transfer_handle<read>
    ttl.wait %xf16 : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL_WRAP-LABEL: func.func @trid_wrap_reuse_write
// TTKERNEL_WRAP-DAG: %[[TRID0:.*]] = arith.constant 0 : i32
// TTKERNEL_WRAP-DAG: %[[NOC:.*]] = arith.constant 0 : i8
// TTKERNEL_WRAP: ttkernel.noc_async_write_set_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL_WRAP: ttkernel.noc_async_write_barrier_with_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL_WRAP: ttkernel.noc_async_write_set_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
module {
  func.func @trid_wrap_reuse_write(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb5 = ttl.bind_cb {cb_index = 5, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb6 = ttl.bind_cb {cb_index = 6, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb7 = ttl.bind_cb {cb_index = 7, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb8 = ttl.bind_cb {cb_index = 8, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb9 = ttl.bind_cb {cb_index = 9, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb10 = ttl.bind_cb {cb_index = 10, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb11 = ttl.bind_cb {cb_index = 11, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb12 = ttl.bind_cb {cb_index = 12, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb13 = ttl.bind_cb {cb_index = 13, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb14 = ttl.bind_cb {cb_index = 14, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb15 = ttl.bind_cb {cb_index = 15, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb16 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf0 = ttl.copy %cb0, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf1 = ttl.copy %cb1, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf2 = ttl.copy %cb2, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf3 = ttl.copy %cb3, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf4 = ttl.copy %cb4, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf5 = ttl.copy %cb5, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf6 = ttl.copy %cb6, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf7 = ttl.copy %cb7, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf8 = ttl.copy %cb8, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf9 = ttl.copy %cb9, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf10 = ttl.copy %cb10, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf11 = ttl.copy %cb11, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf12 = ttl.copy %cb12, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf13 = ttl.copy %cb13, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf14 = ttl.copy %cb14, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf15 = ttl.copy %cb15, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf16 = ttl.copy %cb16, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    ttl.wait %xf0 : !ttl.transfer_handle<write>
    ttl.wait %xf1 : !ttl.transfer_handle<write>
    ttl.wait %xf2 : !ttl.transfer_handle<write>
    ttl.wait %xf3 : !ttl.transfer_handle<write>
    ttl.wait %xf4 : !ttl.transfer_handle<write>
    ttl.wait %xf5 : !ttl.transfer_handle<write>
    ttl.wait %xf6 : !ttl.transfer_handle<write>
    ttl.wait %xf7 : !ttl.transfer_handle<write>
    ttl.wait %xf8 : !ttl.transfer_handle<write>
    ttl.wait %xf9 : !ttl.transfer_handle<write>
    ttl.wait %xf10 : !ttl.transfer_handle<write>
    ttl.wait %xf11 : !ttl.transfer_handle<write>
    ttl.wait %xf12 : !ttl.transfer_handle<write>
    ttl.wait %xf13 : !ttl.transfer_handle<write>
    ttl.wait %xf14 : !ttl.transfer_handle<write>
    ttl.wait %xf15 : !ttl.transfer_handle<write>
    ttl.wait %xf16 : !ttl.transfer_handle<write>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL_WRAP-LABEL: func.func @trid_wrap_reuse_uses_previous_direction
// TTKERNEL_WRAP-DAG: %[[TRID0:.*]] = arith.constant 0 : i32
// TTKERNEL_WRAP-DAG: %[[NOC:.*]] = arith.constant 0 : i8
// TTKERNEL_WRAP: ttkernel.noc_async_write_set_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL_WRAP: ttkernel.noc_async_write_barrier_with_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL_WRAP: ttkernel.noc_async_read_set_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
module {
  func.func @trid_wrap_reuse_uses_previous_direction(%src: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, %dst: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 5 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb5 = ttl.bind_cb {cb_index = 5, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb6 = ttl.bind_cb {cb_index = 6, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb7 = ttl.bind_cb {cb_index = 7, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb8 = ttl.bind_cb {cb_index = 8, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb9 = ttl.bind_cb {cb_index = 9, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb10 = ttl.bind_cb {cb_index = 10, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb11 = ttl.bind_cb {cb_index = 11, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb12 = ttl.bind_cb {cb_index = 12, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb13 = ttl.bind_cb {cb_index = 13, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb14 = ttl.bind_cb {cb_index = 14, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb15 = ttl.bind_cb {cb_index = 15, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb16 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %src_slice = ttl.tensor_slice %src[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %dst_slice = ttl.tensor_slice %dst[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf0 = ttl.copy %cb0, %dst_slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    %xf1 = ttl.copy %src_slice, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf2 = ttl.copy %src_slice, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf3 = ttl.copy %src_slice, %cb3 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf4 = ttl.copy %src_slice, %cb4 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf5 = ttl.copy %src_slice, %cb5 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf6 = ttl.copy %src_slice, %cb6 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf7 = ttl.copy %src_slice, %cb7 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf8 = ttl.copy %src_slice, %cb8 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf9 = ttl.copy %src_slice, %cb9 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf10 = ttl.copy %src_slice, %cb10 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf11 = ttl.copy %src_slice, %cb11 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf12 = ttl.copy %src_slice, %cb12 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf13 = ttl.copy %src_slice, %cb13 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf14 = ttl.copy %src_slice, %cb14 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf15 = ttl.copy %src_slice, %cb15 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf16 = ttl.copy %src_slice, %cb16 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<write>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    ttl.wait %xf2 : !ttl.transfer_handle<read>
    ttl.wait %xf3 : !ttl.transfer_handle<read>
    ttl.wait %xf4 : !ttl.transfer_handle<read>
    ttl.wait %xf5 : !ttl.transfer_handle<read>
    ttl.wait %xf6 : !ttl.transfer_handle<read>
    ttl.wait %xf7 : !ttl.transfer_handle<read>
    ttl.wait %xf8 : !ttl.transfer_handle<read>
    ttl.wait %xf9 : !ttl.transfer_handle<read>
    ttl.wait %xf10 : !ttl.transfer_handle<read>
    ttl.wait %xf11 : !ttl.transfer_handle<read>
    ttl.wait %xf12 : !ttl.transfer_handle<read>
    ttl.wait %xf13 : !ttl.transfer_handle<read>
    ttl.wait %xf14 : !ttl.transfer_handle<read>
    ttl.wait %xf15 : !ttl.transfer_handle<read>
    ttl.wait %xf16 : !ttl.transfer_handle<read>
    func.return
  }
}
