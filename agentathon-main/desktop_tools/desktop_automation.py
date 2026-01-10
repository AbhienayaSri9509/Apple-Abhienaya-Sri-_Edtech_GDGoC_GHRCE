import os
import platform
import time
import math
import warnings

# NOTE: This file contains desktop automation tools provided by the user.
# These cannot be run directly in the web browser environment but are saved here
# for reference or future use in a desktop wrapper/backend.

# Stub for lazy_import since the original codebase likely had this utility
def lazy_import(name):
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError:
        print(f"Warning: Could not import {name}")
        return None

# Lazy import of pyautogui
pyautogui = lazy_import("pyautogui")

# Check if pyautogui is available
if pyautogui is None:
    raise ImportError(
        "pyautogui is required for desktop automation. "
        "Please install it with: pip install pyautogui"
    )

# Lazy import of optional packages
try:
    cv2 = lazy_import("cv2")
except:
    cv2 = None
np = lazy_import("numpy")
plt = lazy_import("matplotlib.pyplot")
try: 
    from PIL import Image 
except: 
    Image = None

try:
    from IPython.display import display
except:
    display = print

try:
    from ..utils.recipient_utils import format_to_recipient
except:
    def format_to_recipient(msg, role):
        return f"[{role}]: {msg}"


class Keyboard:
    """A class to simulate keyboard inputs"""

    def __init__(self, computer):
        self.computer = computer

    def write(self, text, interval=None, delay=0.30, **kwargs):
        """
        Type out a string of characters with some realistic delay.
        """
        time.sleep(delay / 2)

        if interval:
            pyautogui.write(text, interval=interval)
        else:
            try:
                clipboard_history = self.computer.clipboard.view()
            except:
                pass

            ends_in_enter = False

            if text.endswith("\n"):
                ends_in_enter = True
                text = text[:-1]

            lines = text.split("\n")

            if len(lines) < 5:
                for i, line in enumerate(lines):
                    line = line + "\n" if i != len(lines) - 1 else line
                    self.computer.clipboard.copy(line)
                    self.computer.clipboard.paste()
            else:
                # just do it all at once
                self.computer.clipboard.copy(text)
                self.computer.clipboard.paste()

            if ends_in_enter:
                self.press("enter")

            try:
                self.computer.clipboard.copy(clipboard_history)
            except:
                pass

        time.sleep(delay / 2)

    def press(self, *args, presses=1, interval=0.1):
        keys = args
        """
        Press a key or a sequence of keys.

        If keys is a string, it is treated as a single key and is pressed the number of times specified by presses.
        If keys is a list, each key in the list is pressed once.
        """
        time.sleep(0.15)
        pyautogui.press(keys, presses=presses, interval=interval)
        time.sleep(0.15)

    def press_and_release(self, *args, presses=1, interval=0.1):
        """
        Press and release a key or a sequence of keys.

        This method is a perfect proxy for the press method.
        """
        return self.press(*args, presses=presses, interval=interval)

    def hotkey(self, *args, interval=0.1):
        """
        Press a sequence of keys in the order they are provided, and then release them in reverse order.
        """
        time.sleep(0.15)
        modifiers = ["command", "option", "alt", "ctrl", "shift"]
        if "darwin" in platform.system().lower() and len(args) == 2:
            # pyautogui.hotkey seems to not work, so we use applescript
            # Determine which argument is the keystroke and which is the modifier
            keystroke, modifier = (
                args if args[0].lower() not in modifiers else args[::-1]
            )

            modifier = modifier.lower()

            # Map the modifier to the one that AppleScript expects
            if " down" not in modifier:
                modifier = modifier + " down"

            if keystroke.lower() == "space":
                keystroke = " "

            if keystroke.lower() == "enter":
                keystroke = "\n"

            # Create the AppleScript
            script = f"""
            tell application "System Events"
                keystroke "{keystroke}" using {modifier}
            end tell
            """

            # Execute the AppleScript
            os.system("osascript -e '{}'".format(script))
        else:
            pyautogui.hotkey(*args, interval=interval)
        time.sleep(0.15)

    def down(self, key):
        """
        Press down a key.
        """
        time.sleep(0.15)
        pyautogui.keyDown(key)
        time.sleep(0.15)

    def up(self, key):
        """
        Release a key.
        """
        time.sleep(0.15)
        pyautogui.keyUp(key)
        time.sleep(0.15)


class Mouse:
    def __init__(self, computer):
        self.computer = computer

    def scroll(self, clicks):
        """
        Scrolls the mouse wheel up or down the specified number of clicks.
        """
        pyautogui.scroll(clicks)

    def position(self):
        """
        Get the current mouse position.

        Returns:
            tuple: A tuple (x, y) representing the mouse's current position on the screen.
        """
        try:
            return pyautogui.position()
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while retrieving the mouse position: {e}. "
            )

    def move(self, *args, x=None, y=None, icon=None, text=None, screenshot=None):
        """
        Moves the mouse to specified coordinates, an icon, or text.
        """
        if len(args) > 1:
            raise ValueError(
                "Too many positional arguments provided. To move/click specific coordinates, use kwargs (x=x, y=y).\n\nPlease take a screenshot with computer.display.view() to find text/icons to click, then use computer.mouse.click(text) or computer.mouse.click(icon=description_of_icon) if at all possible. This is **significantly** more accurate than using coordinates. Specifying (x=x, y=y) is highly likely to fail. Specifying ('text to click') is highly likely to succeed."
            )
        elif len(args) == 1 or text != None:
            if len(args) == 1:
                text = args[0]

            if screenshot == None:
                screenshot = self.computer.display.screenshot(show=False)

            coordinates = self.computer.display.find(
                '"' + text + '"', screenshot=screenshot
            )

            is_fuzzy = any([c["similarity"] != 1 for c in coordinates])
            # nah just hey, if it's fuzzy, then whatever, it prob wont see the message then decide something else (not really smart enough yet usually)
            # so for now, just lets say it's always not fuzzy so if there's 1 coord it will pick it automatically
            is_fuzzy = False

            if len(coordinates) == 0:
                return self.move(icon=text)  # Is this a better solution?

                if self.computer.emit_images:
                    plt.imshow(np.array(screenshot))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        plt.show()
                raise ValueError(
                    f"@@@HIDE_TRACEBACK@@@Your text ('{text}') was not found on the screen. Please try again. If you're 100% sure the text should be there, consider using `computer.mouse.scroll(-10)` to scroll down.\n\nYou can use `computer.display.get_text_as_list_of_lists()` to see all the text on the screen."
                )
            elif len(coordinates) > 1 or is_fuzzy:
                if self.computer.emit_images:
                    # Convert the screenshot to a numpy array for drawing
                    img_array = np.array(screenshot)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    img_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

                    # Iterate over the response items
                    for i, item in enumerate(coordinates):
                        width, height = screenshot.size
                        x, y = item["coordinates"]
                        x *= width
                        y *= height

                        x = int(x)
                        y = int(y)

                        # Draw a solid blue circle around the found text
                        cv2.circle(img_draw, (x, y), 20, (0, 0, 255), -1)
                        # Put the index number in the center of the circle in white
                        cv2.putText(
                            img_draw,
                            str(i),
                            (x - 10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    img_pil = Image.fromarray(img_draw)
                    display(img_pil)

                coordinates = [
                    f"{i}: ({int(item['coordinates'][0]*self.computer.display.width)}, {int(item['coordinates'][1]*self.computer.display.height)}) "
                    + '"'
                    + item["text"]
                    + '"'
                    for i, item in enumerate(coordinates)
                ]
                if is_fuzzy:
                    error_message = (
                        f"@@@HIDE_TRACEBACK@@@Your text ('{text}') was not found exactly, but some similar text was found. Please review the attached image, then click/move over one of the following coordinates with computer.mouse.click(x=x, y=y) or computer.mouse.move(x=x, y=y):\n"
                        + "\n".join(coordinates)
                    )
                else:
                    error_message = (
                        f"@@@HIDE_TRACEBACK@@@Your text ('{text}') was found multiple times on the screen. Please review the attached image, then click/move over one of the following coordinates with computer.mouse.click(x=x, y=y) or computer.mouse.move(x=x, y=y):\n"
                        + "\n".join(coordinates)
                    )
                raise ValueError(error_message)
            else:
                x, y = coordinates[0]["coordinates"]
                x *= self.computer.display.width
                y *= self.computer.display.height
                x = int(x)
                y = int(y)

        elif x is not None and y is not None:
            print(
                format_to_recipient(
                    "Unless you have just received these EXACT coordinates from a computer.mouse.move or computer.mouse.click command, PLEASE take a screenshot with computer.display.view() to find TEXT OR ICONS to click, then use computer.mouse.click(text) or computer.mouse.click(icon=description_of_icon) if at all possible. This is **significantly** more accurate than using coordinates. Specifying (x=x, y=y) is highly likely to fail. Specifying ('text to click') is highly likely to succeed.",
                    "assistant",
                )
            )
        elif icon is not None:
            if screenshot == None:
                screenshot = self.computer.display.screenshot(show=False)

            coordinates = self.computer.display.find(icon.strip('"'), screenshot)

            if len(coordinates) > 1:
                if self.computer.emit_images:
                    # Convert the screenshot to a numpy array for drawing
                    img_array = np.array(screenshot)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    img_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

                    # Iterate over the response items
                    for i, item in enumerate(coordinates):
                        width, height = screenshot.size
                        x, y = item
                        x *= width
                        y *= height

                        x = int(x)
                        y = int(y)

                        # Draw a solid blue circle around the found text
                        cv2.circle(img_draw, (x, y), 20, (0, 0, 255), -1)
                        # Put the index number in the center of the circle in white
                        cv2.putText(
                            img_draw,
                            str(i),
                            (x - 10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    plt.imshow(img_draw)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        plt.show()

                coordinates = [
                    f"{i}: {int(item[0]*self.computer.display.width)}, {int(item[1]*self.computer.display.height)}"
                    for i, item in enumerate(coordinates)
                ]
                error_message = (
                    f"Your icon ('{text}') was found multiple times on the screen. Please click one of the following coordinates with computer.mouse.move(x=x, y=y):\n"
                    + "\n".join(coordinates)
                )
                raise ValueError(error_message)
            else:
                x, y = coordinates[0]
                x *= self.computer.display.width
                y *= self.computer.display.height
                x = int(x)
                y = int(y)

        else:
            raise ValueError("Either text, icon, or both x and y must be provided")

        if self.computer.verbose:
            if not screenshot:
                screenshot = self.computer.display.screenshot(show=False)

            # Convert the screenshot to a numpy array for drawing
            img_array = np.array(screenshot)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            # Scale drawing_x and drawing_y from screen size to screenshot size for drawing purposes
            drawing_x = int(x * screenshot.width / self.computer.display.width)
            drawing_y = int(y * screenshot.height / self.computer.display.height)

            # Draw a solid blue circle around the place we're clicking
            cv2.circle(img_draw, (drawing_x, drawing_y), 20, (0, 0, 255), -1)

            plt.imshow(img_draw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.show()

        # pyautogui.moveTo(x, y, duration=0.5)
        smooth_move_to(x, y)

    def click(self, *args, button="left", clicks=1, interval=0.1, **kwargs):
        """
        Clicks the mouse at the specified coordinates, icon, or text.
        """
        if args or kwargs:
            self.move(*args, **kwargs)
        pyautogui.click(button=button, clicks=clicks, interval=interval)

    def double_click(self, *args, button="left", interval=0.1, **kwargs):
        """
        Double-clicks the mouse at the specified coordinates, icon, or text.
        """
        if args or kwargs:
            self.move(*args, **kwargs)
        pyautogui.doubleClick(button=button, interval=interval)

    def triple_click(self, *args, button="left", interval=0.1, **kwargs):
        """
        Triple-clicks the mouse at the specified coordinates, icon, or text.
        """
        if args or kwargs:
            self.move(*args, **kwargs)
        pyautogui.tripleClick(button=button, interval=interval)

    def right_click(self, *args, **kwargs):
        """
        Right-clicks the mouse at the specified coordinates, icon, or text.
        """
        if args or kwargs:
            self.move(*args, **kwargs)
        pyautogui.rightClick()

    def down(self):
        """
        Presses the mouse button down.
        """
        pyautogui.mouseDown()

    def up(self):
        """
        Releases the mouse button.
        """
        pyautogui.mouseUp()


def smooth_move_to(x, y, duration=2):
    start_x, start_y = pyautogui.position()
    dx = x - start_x
    dy = y - start_y
    distance = math.hypot(dx, dy)  # Calculate the distance in pixels

    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            break

        t = elapsed_time / duration
        eased_t = (1 - math.cos(t * math.pi)) / 2  # easeInOutSine function

        target_x = start_x + dx * eased_t
        target_y = start_y + dy * eased_t
        pyautogui.moveTo(target_x, target_y)

    # Ensure the mouse ends up exactly at the target (x, y)
    pyautogui.moveTo(x, y)
