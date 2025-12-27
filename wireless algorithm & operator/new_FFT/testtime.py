import numpy as np
import acl
import time

# 宏定义
MODEL_BATCH = 1024
ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 0
ACL_MEMCPY_DEVICE_TO_HOST = 1
ACL_SUCCESS = 0

def check_ret(msg, ret):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{msg} failed ret={ret}")

def run_acl_model(model_id, model_desc, host_x):
    # 记录进入函数的时间
    t_func_start = time.time()

    # 1. [准备阶段] 申请内存 & 搬运数据 (Host -> Device)
    input_size = host_x.size * host_x.itemsize
    input_device, ret = acl.rt.malloc(input_size, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc", ret)
    
    # 兼容性处理：使用 numpy 指针直接拷贝
    if "bytes_to_ptr" in dir(acl.util):
        input_ptr = acl.util.bytes_to_ptr(host_x.tobytes())
    else:
        input_ptr = acl.util.numpy_to_ptr(host_x)
        
    ret = acl.rt.memcpy(input_device, input_size, input_ptr,
                        input_size, ACL_MEMCPY_HOST_TO_DEVICE); check_ret("acl.rt.memcpy", ret)
    
    input_buf = acl.create_data_buffer(input_device, input_size)
    input_ds = acl.mdl.create_dataset()
    _, ret = acl.mdl.add_dataset_buffer(input_ds, input_buf); check_ret("add_dataset_buffer", ret)

    # 2. [输出准备]
    out_num = acl.mdl.get_num_outputs(model_desc)
    output_ds = acl.mdl.create_dataset()
    out_dev, out_sizes, out_bufs = [], [], []
    for i in range(out_num):
        size_i = acl.mdl.get_output_size_by_index(model_desc, i)
        dev_i, ret = acl.rt.malloc(size_i, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc", ret)
        buf_i = acl.create_data_buffer(dev_i, size_i)
        _, ret = acl.mdl.add_dataset_buffer(output_ds, buf_i); check_ret("add_dataset_buffer", ret)
        out_dev.append(dev_i); out_sizes.append(size_i); out_bufs.append(buf_i)

    # =========================================================
    # 3. [核心推理] 计时重点
    # =========================================================
    t_infer_start = time.time()
    
    # 同步执行，直到 NPU 算完才返回
    ret = acl.mdl.execute(model_id, input_ds, output_ds)
    check_ret("acl.mdl.execute", ret)
    
    t_infer_end = time.time()
    # =========================================================

    # 4. [结果回传] (Device -> Host)
    host_out = np.zeros(host_x.shape, dtype=np.float32)
    ret = acl.rt.memcpy(host_out.ctypes.data, out_sizes[0],
                        out_dev[0], out_sizes[0], ACL_MEMCPY_DEVICE_TO_HOST); check_ret("acl.rt.memcpy D2H", ret)

    # 5. [清理资源]
    ret = acl.rt.free(input_device)
    for dev, buf in zip(out_dev, out_bufs):
        ret = acl.rt.free(dev)
        ret = acl.destroy_data_buffer(buf)
    ret = acl.mdl.destroy_dataset(input_ds)
    ret = acl.mdl.destroy_dataset(output_ds)
    ret = acl.destroy_data_buffer(input_buf)

    t_func_end = time.time()
    
    # 计算耗时 (毫秒)
    infer_cost = (t_infer_end - t_infer_start) * 1000
    total_cost = (t_func_end - t_func_start) * 1000
    
    return host_out, infer_cost, total_cost

if __name__ == "__main__":
    # ... (前面的 acl.init, set_device, load_model 代码保持不变) ...
    # 为了节省篇幅，假设你已经完成了 acl.init 到 acl.mdl.load_from_file 的部分
    # 这里直接从准备数据开始
    
    print("[Info] ACL init success. Loading model...")
    # 请确保以下变量已初始化: model_id, model_desc
    # -----------------------------------------------------------
    # 补全初始化代码以便你能直接复制运行 (假设 model_id 等已存在)
    ret = acl.init()
    ret = acl.rt.set_device(0)
    context, ret = acl.rt.create_context(0)
    stream, ret = acl.rt.create_stream()
    model_id, ret = acl.mdl.load_from_file("dft256_mat_1024.om") # 确保文件名正确
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id)
    # -----------------------------------------------------------

    # 准备测试数据
    np.random.seed(42)
    x_batch = np.random.randn(MODEL_BATCH, 2, 256).astype(np.float32)

    print("\n" + "="*60)
    print(f"{'Count':<5} | {'Inference(ms)':<15} | {'Total(ms) (w/ Mem)':<20}")
    print("-" * 60)

    # 循环测试 10 次
    for i in range(10):
        out, t_infer, t_total = run_acl_model(model_id, model_desc, x_batch)
        print(f"{i:<5} | {t_infer:<15.3f} | {t_total:<20.3f}")
        
        # 稍微 sleep 一下模拟真实业务间隔，避免日志刷新太快
        # time.sleep(0.01) 

    print("="*60 + "\n")
    print("注意：第一次调用通常较慢（Warmup），后续才是真实性能。")
    print("Total(ms) 包含了 malloc/free 的时间，这在高性能场景应优化掉。")

    # ... (后面的 unload, destroy 资源释放代码保持不变) ...
    ret = acl.mdl.unload(model_id)
    ret = acl.mdl.destroy_desc(model_desc)
    ret = acl.rt.destroy_stream(stream)
    ret = acl.rt.destroy_context(context)
    ret = acl.rt.reset_device(0)
    ret = acl.finalize()