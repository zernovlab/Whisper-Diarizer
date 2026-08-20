"""Entry point: launches the desktop GUI."""
import sys

from PySide6.QtWidgets import QApplication

from app.gui import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
