import urllib.request
import bz2
import os

def install_openh264():
    url = "http://ciscobinary.openh264.org/openh264-1.8.0-win64.dll.bz2"
    filename = "openh264-1.8.0-win64.dll.bz2"
    target_dll = "openh264-1.8.0-win64.dll"
    
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
        
        print(f"Extracting to {target_dll}...")
        with bz2.BZ2File(filename, 'rb') as f_in:
            with open(target_dll, 'wb') as f_out:
                f_out.write(f_in.read())
        
        print("Extraction complete.")
        
        # Cleanup
        os.remove(filename)
        print("Cleaned up temporary file.")
        
        print(f"Success! {target_dll} is ready. Please restart your python server.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    install_openh264()
