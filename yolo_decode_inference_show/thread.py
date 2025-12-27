# ==================== 全局控制 ====================
stop_event = threading.Event()

def signal_handler(sig, frame):
    print("\n收到中断信号，正在退出...")
    stop_event.set()
    cv2.destroyAllWindows()
    sys.exit(0)

# ==================== 主入口 ====================
def main():
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    print("="*70)
    print("启动双线程")
    print(f"监控目录: {TEMP_DIR}")
    print(f"文件ID范围: 0-{MAX_FILE_ID} (循环)")
    print("="*70)
    
    # 创建两个线程
    thread1 = threading.Thread(target=task1, name="Producer")   # 把rx的主函数改个名称，替换task1就行，rx保存文件时按0-9.h264循环保存
    thread2 = threading.Thread(target=inference_and_show, name="Consumer")
    
    # 设置为守护线程（主线程退出时自动结束）
    thread1.daemon = True
    thread2.daemon = True
    
    # 启动线程
    thread1.start()
    thread2.start()
    
    # 等待两个线程结束（或被信号中断）
    try:
        while thread1.is_alive() or thread2.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    
    print("主程序退出")

if __name__ == "__main__":
    main()
