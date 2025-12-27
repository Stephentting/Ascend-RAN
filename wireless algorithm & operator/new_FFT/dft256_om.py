import numpy as np
import acl
import time

# 宏定义：.om 模型编译时的 batch 数,注意这个是编译静态图时就已经确定了，该数字要换模型文件
MODEL_BATCH = 1024
ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 0
ACL_MEMCPY_DEVICE_TO_HOST = 1
ACL_SUCCESS = 0

def check_ret(msg, ret):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{msg} failed ret={ret}")

def run_acl_model(model_id, model_desc, host_x):
    # 输入数据如何得到：此处使用随机模拟 (MODEL_BATCH, 2, 256) 的实虚部
    input_size = host_x.size * host_x.itemsize
    input_device, ret = acl.rt.malloc(input_size, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc", ret)
    ret = acl.rt.memcpy(input_device, input_size, acl.util.bytes_to_ptr(host_x.tobytes()),
                        input_size, ACL_MEMCPY_HOST_TO_DEVICE); check_ret("acl.rt.memcpy", ret)
    input_buf = acl.create_data_buffer(input_device, input_size)
    input_ds = acl.mdl.create_dataset()
    _, ret = acl.mdl.add_dataset_buffer(input_ds, input_buf); check_ret("acl.mdl.add_dataset_buffer", ret)

    # 输出数据是怎样的：IDFT 输出同样为 (MODEL_BATCH, 2, 256) 的实虚部 float32
    out_num = acl.mdl.get_num_outputs(model_desc)
    output_ds = acl.mdl.create_dataset()
    out_dev, out_sizes, out_bufs = [], [], []
    for i in range(out_num):
        size_i = acl.mdl.get_output_size_by_index(model_desc, i)
        dev_i, ret = acl.rt.malloc(size_i, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc", ret)
        buf_i = acl.create_data_buffer(dev_i, size_i)
        _, ret = acl.mdl.add_dataset_buffer(output_ds, buf_i); check_ret("acl.mdl.add_dataset_buffer", ret)
        out_dev.append(dev_i); out_sizes.append(size_i); out_bufs.append(buf_i)

    # 如何执行模型推理：执行 acl.mdl.execute 并计时
    start = time.time()
    ret = acl.mdl.execute(model_id, input_ds, output_ds); check_ret("acl.mdl.execute", ret)
    elapsed = time.time() - start

    host_out = np.empty(host_x.shape, dtype=np.float32)
    ret = acl.rt.memcpy(acl.util.bytes_to_ptr(host_out.tobytes()), out_sizes[0],
                        out_dev[0], out_sizes[0], ACL_MEMCPY_DEVICE_TO_HOST); check_ret("acl.rt.memcpy D2H", ret)

    # 释放资源
    ret = acl.rt.free(input_device); check_ret("acl.rt.free input", ret)
    for dev, buf in zip(out_dev, out_bufs):
        ret = acl.rt.free(dev); check_ret("acl.rt.free output", ret)
        ret = acl.destroy_data_buffer(buf); check_ret("acl.destroy_data_buffer output", ret)
    ret = acl.mdl.destroy_dataset(input_ds); check_ret("acl.mdl.destroy_dataset input", ret)
    ret = acl.mdl.destroy_dataset(output_ds); check_ret("acl.mdl.destroy_dataset output", ret)
    ret = acl.destroy_data_buffer(input_buf); check_ret("acl.destroy_data_buffer input", ret)
    return host_out, elapsed

if __name__ == "__main__":
    # 随机模拟输入
    np.random.seed(42)
    x_batch = np.random.randn(MODEL_BATCH, 2, 256).astype(np.float32)

    ret = acl.init(); check_ret("acl.init", ret)
    dev_id = 0
    ret = acl.rt.set_device(dev_id); check_ret("acl.rt.set_device", ret)
    ctx, ret = acl.rt.create_context(dev_id); check_ret("acl.rt.create_context", ret)
    stream, ret = acl.rt.create_stream(); check_ret("acl.rt.create_stream", ret)

    # 根据实际选择对应的 .om 模型文件，idft就用idft的模型，反正输入输出是一样的
    model_id, ret = acl.mdl.load_from_file("dft256_mat_1024.om"); check_ret("acl.mdl.load_from_file", ret)
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id); check_ret("acl.mdl.get_desc", ret)

#####下面是推理，只要循环运行下面这个就行，不需要每次都load和unload模型
#测试没问题就把run_acl_model计时功能删掉吧
    out, t = run_acl_model(model_id, model_desc, x_batch)
    print(f"idft256_mat.om batch={MODEL_BATCH} 输出形状 {out.shape}, 总耗时 {t*1000:.2f} ms")
#####上面是推理
    ret = acl.mdl.unload(model_id); check_ret("acl.mdl.unload", ret)
    ret = acl.mdl.destroy_desc(model_desc); check_ret("acl.mdl.destroy_desc", ret)
    ret = acl.rt.destroy_stream(stream); check_ret("acl.rt.destroy_stream", ret)
    ret = acl.rt.destroy_context(ctx); check_ret("acl.rt.destroy_context", ret)
    ret = acl.rt.reset_device(dev_id); check_ret("acl.rt.reset_device", ret)
    ret = acl.finalize(); check_ret("acl.finalize", ret)
