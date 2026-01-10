import time
import sys
import pyautogui
# Attempt to import pyperclip for clipboard operations (needed for Keyboard class)
try:
    import pyperclip
except ImportError:
    pyperclip = None
    print("Warning: 'pyperclip' not installed. Clipboard operations main not work fully.")

# Import the classes from the sibling file
try:
    # Try relative import first (when used as a package)
    from .desktop_automation import Keyboard, Mouse
except ImportError:
    # Fall back to absolute import (when run as a script)
    from desktop_automation import Keyboard, Mouse

class MockClipboard:
    def view(self):
        if pyperclip:
            return pyperclip.paste()
        return ""
    
    def copy(self, text):
        if pyperclip:
            pyperclip.copy(text)
        else:
            print(f"[MockClipboard] Copy: {text}")

    def paste(self):
        # In a real scenario, this might trigger a Ctrl+V hotkey, 
        # but the Keyboard class seems to use this to 'simulate' pasting
        # or just manipulating the clipboard content before typing?
        # The Keyboard.write method actually calls self.computer.clipboard.paste() 
        # which presumably triggers the paste action.
        # For this test, we might just print usage or actually hotkey.
        pass

class MockDisplay:
    def __init__(self):
        self.width, self.height = pyautogui.size()

    def screenshot(self, show=False):
        return pyautogui.screenshot()

    def find(self, text, screenshot=None):
        # Stub: Real text finding requires OCR (Tesseract/etc).
        # We will return empty or a dummy coord to prevent crashing if tested.
        print(f"[MockDisplay] Requested to find text: {text}")
        return []

class MockComputer:
    def __init__(self):
        self.clipboard = MockClipboard()
        self.display = MockDisplay()
        self.emit_images = False
        self.verbose = True

def run_checks():
    print("Initializing Mock Computer Environment...")
    computer = MockComputer()
    
    print("Initializing Keyboard and Mouse tools...")
    keyboard = Keyboard(computer)
    mouse = Mouse(computer)

    print("\n--- TEST 1: MOUSE ---")
    print("Moving mouse in a small square starting in 3 seconds.")
    print("Please watch your cursor.")
    time.sleep(3)
    
    start_x, start_y = pyautogui.position()
    offset = 100
    
    # Move Right
    print("Moving Right...")
    mouse.move(x=start_x + offset, y=start_y)
    
    # Move Down
    print("Moving Down...")
    mouse.move(x=start_x + offset, y=start_y + offset)
    
    # Move Left
    print("Moving Left...")
    mouse.move(x=start_x, y=start_y + offset)
    
    # Move Up (Original)
    print("Moving Up...")
    mouse.move(x=start_x, y=start_y)
    
    print("Mouse check complete.")

    print("\n--- TEST 2: KEYBOARD ---")
    print("We will type 'Hello from Lumina' in 5 seconds.")
    print(">>> PLEASE CLICK INTO A TEXT EDITOR OR INPUT FIELD NOW <<<")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    # We use interval to type slowly so user can see it
    keyboard.write("Hello from Lumina!", interval=0.1)
    
    print("\n\n--- TEST 3: HOTKEY ---")
    print("Testing 'Ctrl' key press (should be harmless) in 2 seconds...")
    time.sleep(2)
    keyboard.press('ctrl')
    
    print("\nChecks finished successfully.")

if __name__ == "__main__":
    # Ensure dependencies are met
    if 'pyautogui' not in sys.modules:
        import pyautogui
    
    try:
        run_checks()
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
