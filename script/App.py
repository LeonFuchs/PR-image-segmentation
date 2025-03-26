import sys
from PyQt6.QtWidgets import QApplication
from GUI import Gui

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Gui()
    window.show()
    sys.exit(app.exec())