#!/usr/bin/env python3
"""
Aurora utility functions
"""
import threading 
import queue as queue_module
from aurora.config import SHUTDOWN_EVENT

def get_input_with_shutdown_check(prompt, timeout=0.5):
    """Get user input while checking for shutdown events."""
    import sys
    
    def input_thread(input_queue):
        try:
            user_input = input(prompt)
            input_queue.put(user_input)
        except (EOFError, KeyboardInterrupt):
            input_queue.put("__INTERRUPT__")
        except Exception as e:
            input_queue.put("__ERROR__")
    
    input_queue = queue_module.Queue()
    thread = threading.Thread(target=input_thread, args=(input_queue,), daemon=True)
    thread.start()
    
    # Wait for input or shutdown
    while thread.is_alive():
        if SHUTDOWN_EVENT.is_set():
            return "__SHUTDOWN__"
        
        try:
            # Check if input is ready
            result = input_queue.get(timeout=timeout)
            return result
        except queue_module.Empty:
            continue
    
    # Thread finished, get the result
    try:
        return input_queue.get_nowait()
    except queue_module.Empty:
        return "__TIMEOUT__"
