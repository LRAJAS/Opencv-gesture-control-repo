import HandTrackingModule as htm

def main():
    try:
        detector = htm.handDetector()
        print("handDetector initialized successfully!")
    except AttributeError as e:
        print(f"AttributeError: {e}")
    except ImportError as e:
        print(f"ImportError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
