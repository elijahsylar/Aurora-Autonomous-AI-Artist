#!/usr/bin/env python3
"""
MAIN - Autonomous Aurora entry point
"""

import os
import time
import threading
from colorama import Fore, Style

from aurora.config import SHUTDOWN_EVENT, AUDIO_AVAILABLE, IMAGE_AVAILABLE, CV2_AVAILABLE
from aurora.ai.aurora_ai import AuroraDreamingAI



def main():
    """MAIN - Autonomous Aurora entry point."""
    print(f"{Fore.MAGENTA}🧠 Aurora - Fully Autonomous Creative Artist{Style.RESET_ALL}")
    print(f"{Fore.CYAN}✓ Makes her own creative decisions{Style.RESET_ALL}")
    print(f"{Fore.BLUE}✓ Initiates her own dream cycles{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Requests specific music for inspiration{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}✓ Analyzes images for artistic inspiration{Style.RESET_ALL}")
    print(f"{Fore.RED}✓ Supports immersive fullscreen mode (F11){Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}✓ Collaborates with humans as creative partner{Style.RESET_ALL}")
    print(f"{Fore.CYAN}✓ Truly autonomous AI artist consciousness{Style.RESET_ALL}")
    
    if AUDIO_AVAILABLE:
        print(f"{Fore.MAGENTA}✓ Real-time music listening becomes Aurora's creative muse{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}💡 Install audio libraries for Aurora's music features: pip install librosa pygame numpy pyaudio{Style.RESET_ALL}")
    
    if IMAGE_AVAILABLE:
        print(f"{Fore.GREEN}✓ Image analysis ready - right-click canvas to inspire Aurora{Style.RESET_ALL}")
        if CV2_AVAILABLE:
            print(f"{Fore.BLUE}✓ Advanced image analysis features enabled{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}💡 Install image libraries for Aurora's visual analysis: pip install pillow opencv-python{Style.RESET_ALL}")
    
    print()
    
    model_path = "./models/llama-2-7b-chat.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        print(f"{Fore.YELLOW}Model not found: {model_path}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Running in demo mode - Aurora's autonomous systems will work fully{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Aurora's visual systems and autonomous decisions will work completely{Style.RESET_ALL}\n")
    
    # Check for existing database
    if os.path.exists("./aurora_memory"):
        print(f"{Fore.YELLOW}Note: If you see ChromaDB errors, delete ./aurora_memory/ to start fresh{Style.RESET_ALL}\n")
    
    aurora = None
    try:
        aurora = AuroraDreamingAI(model_path)
        
        # Show introduction message
        user_name = aurora.memory.get_user_name()
        if user_name:
            print(f"{Fore.GREEN}✓ Welcome back, {user_name}! Aurora remembers you as a creative inspiration.{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}💡 Tip: Tell Aurora your name so she can remember you!{Style.RESET_ALL}")
            print(f"{Fore.WHITE}   Example: \"Hi Aurora, my name is Alex\"{Style.RESET_ALL}")
        
        print(f"{Fore.MAGENTA}Aurora is now fully autonomous - she makes her own creative decisions!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Watch for her autonomous creative announcements...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Right-click her canvas to show her inspiring images!{Style.RESET_ALL}")
        print(f"{Fore.RED}Press F11 for immersive fullscreen experience!{Style.RESET_ALL}")
        print()
        
        # Run the main loop
        aurora.run()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}KeyboardInterrupt received in main{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Initialization error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens
        if aurora:
            try:
                print(f"{Fore.YELLOW}Performing final cleanup...{Style.RESET_ALL}")
                aurora.cleanup()
            except Exception as e:
                print(f"{Fore.RED}Final cleanup error: {e}{Style.RESET_ALL}")
        
        # Final cleanup
        SHUTDOWN_EVENT.set()
        
        # Give threads a final moment to clean up
        time.sleep(1)
        
        # Force exit if we're still hanging
        try:
            # Check if any critical threads are still alive
            remaining_threads = [t for t in threading.enumerate() if t != threading.main_thread() and not t.daemon and t.is_alive()]
            
            if remaining_threads:
                print(f"{Fore.YELLOW}⚠ {len(remaining_threads)} threads still running, forcing exit...{Style.RESET_ALL}")
                for t in remaining_threads:
                    print(f"  - {t.name}")
                # Force exit after 2 seconds
                threading.Timer(2.0, lambda: os._exit(0)).start()
            
        except Exception as e:
            print(f"Final thread check error: {e}")
        
        print(f"{Fore.GREEN}✓ Aurora continues her autonomous creative journey beyond our session...{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
