import os

def run_script(path):
    print(f"\n➡️ Running: {path}\n")
    os.system(f"python \"{path}\"")

menu = {
    "1": ("Hand Tracking Module", "Open CV Projects/Hand-tracking-module/P1_openCV_Hand-tracking-Module.py"),
    "2": ("Virtual Mouse", "Open CV Projects/Virtual-mouse-module/P6_Ai_virtual-mouse.py"),
    "3": ("Volume Control", "Open CV Projects/Volume-control-module/P4_volume-control.py"),
    "4": ("AI Painter", "Open CV Projects/Virtual-Painter-module/P7_Ai-painter_vr.py"),
    "5": ("Face Recognition", "Open CV Projects/Face-recognition-module/P3_face-recognizer.py"),
    "6": ("Virtual Fitness Trainer", "Open CV Projects/Virtual-Fitness-partner-module/P3_CNN-module.py"),
    "7": ("Basic Camera Test", "Basic Testing/camera-test.py"),
    "8": ("Experimental OpenCV Script", "Opencv Exp/openCV_1_(camera-open).py"),
    "0": ("Exit", None)
}

def main():
    while True:
        print("\n🖐️ OpenCV Gesture Control Launcher\n")
        for key, (name, _) in menu.items():
            print(f"{key}. {name}")

        choice = input("\nEnter your choice: ").strip()

        if choice == "0":
            print("Exiting... 👋")
            break
        elif choice in menu:
            _, script_path = menu[choice]
            run_script(script_path)
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
