import numpy as np
import acl
import time
import cv2
import ctypes
import subprocess
import os

# ==================== 配置参数 ====================
MODEL_PATH = "./person_yolo11n.om"
TEMP_DIR = "./temp"
MAX_FILE_ID = 9
MODEL_INPUT_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
ACL_SUCCESS = 0

# ==================== ACL工具函数 ====================
def check_ret(msg, ret):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{msg} failed ret={ret}")

def create_io_resources(model_desc, input_shape):
    input_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
    input_device, ret = acl.rt.malloc(input_size, 0); check_ret("acl.rt.malloc input", ret)
    input_buf = acl.create_data_buffer(input_device, input_size)
    input_ds = acl.mdl.create_dataset()
    _, ret = acl.mdl.add_dataset_buffer(input_ds, input_buf); check_ret("add_dataset_buffer input", ret)

    output_ds = acl.mdl.create_dataset()
    out_dev, out_sizes, out_bufs = [], [], []
    for i in range(acl.mdl.get_num_outputs(model_desc)):
        size_i = acl.mdl.get_output_size_by_index(model_desc, i)
        dev_i, ret = acl.rt.malloc(size_i, 0); check_ret("acl.rt.malloc output", ret)
        buf_i = acl.create_data_buffer(dev_i, size_i)
        _, ret = acl.mdl.add_dataset_buffer(output_ds, buf_i); check_ret("add_dataset_buffer output", ret)
        out_dev.append(dev_i); out_sizes.append(size_i); out_bufs.append(buf_i)

    return {
        "input_device": input_device, "input_size": input_size, "input_buf": input_buf, "input_ds": input_ds,
        "out_dev": out_dev, "out_sizes": out_sizes, "out_bufs": out_bufs, "output_ds": output_ds,
    }

def destroy_io_resources(io_res):
    acl.rt.free(io_res["input_device"])
    acl.destroy_data_buffer(io_res["input_buf"])
    acl.mdl.destroy_dataset(io_res["input_ds"])
    for dev, buf in zip(io_res["out_dev"], io_res["out_bufs"]):
        acl.rt.free(dev)
        acl.destroy_data_buffer(buf)
    acl.mdl.destroy_dataset(io_res["output_ds"])

# ==================== YOLO推理函数 ====================
def preprocess(img_np):
    h, w = img_np.shape[:2]
    scale = min(MODEL_INPUT_SIZE / h, MODEL_INPUT_SIZE / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img_np, (new_w, new_h))
    padded = np.full((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3), 114, dtype=np.uint8)
    y_off = (MODEL_INPUT_SIZE - new_h) // 2
    x_off = (MODEL_INPUT_SIZE - new_w) // 2
    padded[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2,0,1)[np.newaxis].astype(np.float32) / 255.0

def run_acl_model(model_id, model_desc, host_x, io_res):
    ret = acl.rt.memcpy(io_res["input_device"], io_res["input_size"],
                        acl.util.bytes_to_ptr(host_x.tobytes()),
                        io_res["input_size"], 0); check_ret("acl.rt.memcpy H2D", ret)
    ret = acl.mdl.execute(model_id, io_res["input_ds"], io_res["output_ds"]); check_ret("acl.mdl.execute", ret)
    
    out_size = io_res["out_sizes"][0]
    host_out_buf = ctypes.create_string_buffer(out_size)
    host_out_ptr = ctypes.addressof(host_out_buf)
    ret = acl.rt.memcpy(host_out_ptr, out_size, io_res["out_dev"][0],
                        out_size, 1); check_ret("acl.rt.memcpy D2H", ret)
    host_out_bytes = acl.util.ptr_to_bytes(host_out_ptr, out_size)
    elem_cnt = out_size // np.dtype(np.float32).itemsize
    shape = (1, 5, 8400) if elem_cnt == 5 * 8400 else (1, elem_cnt)
    return np.frombuffer(host_out_bytes, dtype=np.float32).reshape(shape)

def postprocess(pred):
    arr = pred[0].squeeze(0) if pred[0].ndim == 3 else pred[0]
    if arr.shape == (5, 8400):
        arr = arr.transpose(1, 0)
    
    conf_mask = arr[:, 4] > CONF_THRESH
    detections = []
    for i in range(arr.shape[0]):
        if conf_mask[i]:
            cx, cy, w, h = arr[i, :4]
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            detections.append([x1, y1, x2, y2, float(arr[i, 4]), 0])
    
    boxes = [d[:4] for d in detections]
    scores = [d[4] for d in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH)
    if len(indices) == 0:
        return []
    indices = indices.flatten() if isinstance(indices, np.ndarray) else [i[0] for i in indices]
    return [detections[i] for i in indices]

def draw_boxes(img, detections):
    for det in detections:
        x1, y1, x2, y2, score, _ = det
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"person {score:.2f}", (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img

# ==================== H.264解码函数 ====================
def decode_h264_frames(input_file: str):
    """硬件加速解码H.264"""
    # 检测硬件加速
    try:
        hw_accelerations = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True, timeout=3).stdout.lower()
        has_hw = any(x in hw_accelerations for x in ['v4l2_m2m', 'drm', 'vaapi'])
    except:
        has_hw = False
    
    cap = cv2.VideoCapture(input_file, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开: {input_file}")
    
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    print(f"[{'硬件' if has_hw else '软件'}解码] {os.path.basename(input_file)}: {len(frames)}帧, FPS: {fps}")
    return frames, fps

def wait_for_next_file(current_id: int, max_id: int, temp_dir: str, 
                       current_file_to_delete: str = None, poll_interval: float = 0.05):
    """轮询等待下一个文件就绪，并删除当前文件"""
    next_id = (current_id + 1) % (max_id + 1)
    filename = os.path.join(temp_dir, f"{next_id}.h264")
    
    while True:
        if os.path.exists(filename):
            size1 = os.path.getsize(filename)
            time.sleep(poll_interval)
            if os.path.exists(filename) and size1 == os.path.getsize(filename) and size1 > 0:
                if current_file_to_delete and os.path.exists(current_file_to_delete):
                    try:
                        os.remove(current_file_to_delete)
                        print(f"[删除] {os.path.basename(current_file_to_delete)}")
                    except:
                        pass
                return next_id, filename
        time.sleep(poll_interval)

def load_and_process_file(filepath: str, model_id, model_desc, io_res) -> tuple:
    """加载并处理H.264文件"""
    try:
        frames, fps = decode_h264_frames(filepath)
        if not frames:
            return None, None, None
        
        t0 = time.time()
        pred = run_acl_model(model_id, model_desc, preprocess(frames[0]), io_res)
        detections = postprocess(pred)
        print(f"[推理] {len(detections)}目标, {(time.time()-t0)*1000:.1f}ms")
        
        return frames, detections, fps
        
    except Exception as e:
        print(f"[失败] {filepath}: {e}")
        return None, None, None

# ==================== 主流程 ====================
def inference_and_show():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    print(f"监控: {TEMP_DIR}, 范围: 0-{MAX_FILE_ID}, 处理完删除")
    
    # 初始化ACL
    ret = acl.init(); check_ret("acl.init", ret)
    dev_id = 0; acl.rt.set_device(dev_id)
    model_id, _ = acl.mdl.load_from_file(MODEL_PATH)
    model_desc = acl.mdl.create_desc(); acl.mdl.get_desc(model_desc, model_id)
    
    io_res = create_io_resources(model_desc, preprocess(np.zeros((480, 640, 3), dtype=np.uint8)).shape)
    
    current_file_id = -1
    current_file_path = None
    frames_buffer, detections_buffer = [], []
    fps, frame_delay = 30, 33
    
    while True:
        next_id, next_file = wait_for_next_file(current_file_id, MAX_FILE_ID, TEMP_DIR, current_file_path)
        if next_id is None:
            break
        
        if next_id != current_file_id:
            new_frames, new_detections, new_fps = load_and_process_file(next_file, model_id, model_desc, io_res)
            if new_frames is not None:
                frames_buffer, detections_buffer = new_frames, new_detections
                fps, frame_delay = new_fps if new_fps > 0 else 30, int(1000 / (new_fps if new_fps > 0 else 30))
                current_file_id, current_file_path = next_id, next_file
        
        if frames_buffer:
            for idx, frame_bgr in enumerate(frames_buffer):
                frame_with_boxes = draw_boxes(frame_bgr.copy(), detections_buffer)
                cv2.putText(frame_with_boxes, f"File: {current_file_id} Frame: {idx}/{len(frames_buffer)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Live Video", frame_with_boxes)
                
                if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                    destroy_io_resources(io_res)
                    acl.mdl.unload(model_id); acl.mdl.destroy_desc(model_desc)
                    acl.rt.reset_device(dev_id); acl.finalize()
                    cv2.destroyAllWindows()
                    return
    
    destroy_io_resources(io_res)
    acl.mdl.unload(model_id); acl.mdl.destroy_desc(model_desc)
    acl.rt.reset_device(dev_id); acl.finalize()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference_and_show()