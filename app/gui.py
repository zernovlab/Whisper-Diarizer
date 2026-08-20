"""PySide6 desktop UI for the Whisper + diarization pipeline."""
from __future__ import annotations

import html
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app import export
from app.config import load_config, save_config
from app.merge import Chunk
from app.pipeline import PipelineConfig, run_pipeline

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "distil-large-v3"]
LANGUAGES = [
    ("auto", "Автоопределение"),
    ("ru", "Русский"),
    ("en", "English"),
    ("uk", "Українська"),
    ("de", "Deutsch"),
    ("fr", "Français"),
    ("es", "Español"),
]
DEVICES = ["auto", "cuda", "cpu"]
SPEAKER_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706", "#0891b2", "#be185d", "#4b5563"]

STATUS_PENDING = "ожидает"
STATUS_RUNNING = "обработка"
STATUS_DONE = "готово"
STATUS_ERROR = "ошибка"


@dataclass
class QueueItem:
    path: str
    status: str = STATUS_PENDING
    chunks: list[Chunk] = field(default_factory=list)
    error: str = ""


class QueueWorker(QThread):
    item_started = Signal(int)
    item_progress = Signal(int, int, str)
    item_finished = Signal(int, list)
    item_failed = Signal(int, str)
    queue_finished = Signal()

    def __init__(self, jobs: list[tuple[int, PipelineConfig]]):
        super().__init__()
        self.jobs = jobs
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        for index, config in self.jobs:
            if self._stop_requested:
                break
            self.item_started.emit(index)
            try:
                chunks = run_pipeline(
                    config,
                    progress=lambda p, m, idx=index: self.item_progress.emit(idx, p, m),
                )
                self.item_finished.emit(index, chunks)
            except Exception as exc:  # noqa: BLE001
                self.item_failed.emit(index, f"{exc}\n\n{traceback.format_exc()}")
        self.queue_finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознавание речи со спикерами (Whisper)")
        self.resize(950, 900)

        self.cfg = load_config()
        self.queue: list[QueueItem] = []
        self.current_item_index: int | None = None
        self.worker: QueueWorker | None = None
        self.rename_edits: dict[str, QLineEdit] = {}

        self._build_ui()
        self._load_settings_into_ui()

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        # Queue
        layout.addWidget(QLabel("Очередь файлов:"))
        queue_btn_row = QHBoxLayout()
        self.add_files_btn = QPushButton("Добавить файлы...")
        self.add_files_btn.clicked.connect(self.on_add_files)
        self.remove_btn = QPushButton("Убрать выбранное")
        self.remove_btn.clicked.connect(self.on_remove_selected)
        self.clear_btn = QPushButton("Очистить очередь")
        self.clear_btn.clicked.connect(self.on_clear_queue)
        queue_btn_row.addWidget(self.add_files_btn)
        queue_btn_row.addWidget(self.remove_btn)
        queue_btn_row.addWidget(self.clear_btn)
        layout.addLayout(queue_btn_row)

        self.queue_list = QListWidget()
        self.queue_list.setFixedHeight(120)
        self.queue_list.currentRowChanged.connect(self.on_queue_item_selected)
        layout.addWidget(self.queue_list)

        # Settings
        settings_box = QGroupBox("Настройки")
        form = QFormLayout(settings_box)
        self.settings_form = form

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_SIZES)
        form.addRow("Модель Whisper:", self.model_combo)

        self.lang_combo = QComboBox()
        for code, name in LANGUAGES:
            self.lang_combo.addItem(name, userData=code)
        form.addRow("Язык:", self.lang_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItems(DEVICES)
        form.addRow("Устройство:", self.device_combo)

        self.diarization_checkbox = QCheckBox("Включить диаризацию (разделение по спикерам)")
        self.diarization_checkbox.setChecked(True)
        self.diarization_checkbox.toggled.connect(self._update_diarization_fields_visibility)
        form.addRow("", self.diarization_checkbox)

        self.speakers_mode_combo = QComboBox()
        self.speakers_mode_combo.addItem("Определить автоматически", userData="auto")
        self.speakers_mode_combo.addItem("Точное число спикеров", userData="exact")
        self.speakers_mode_combo.addItem("Диапазон число спикеров", userData="range")
        self.speakers_mode_combo.currentIndexChanged.connect(self._update_speaker_fields_visibility)
        form.addRow("Спикеры:", self.speakers_mode_combo)

        self.num_speakers_spin = QSpinBox()
        self.num_speakers_spin.setRange(1, 20)
        form.addRow("Число спикеров:", self.num_speakers_spin)

        range_row = QHBoxLayout()
        self.min_speakers_spin = QSpinBox()
        self.min_speakers_spin.setRange(1, 20)
        self.max_speakers_spin = QSpinBox()
        self.max_speakers_spin.setRange(1, 20)
        range_row.addWidget(QLabel("от"))
        range_row.addWidget(self.min_speakers_spin)
        range_row.addWidget(QLabel("до"))
        range_row.addWidget(self.max_speakers_spin)
        self.range_row_widget = QWidget()
        self.range_row_widget.setLayout(range_row)
        form.addRow("Диапазон:", self.range_row_widget)

        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_edit.setPlaceholderText("hf_...")
        form.addRow("Hugging Face токен:", self.hf_token_edit)

        self.hint_label = QLabel(
            "Нужен бесплатный токен (тип Read, или fine-grained с правом "
            '"Read access to contents of public gated repos"): '
            '<a href="https://huggingface.co/settings/tokens">huggingface.co/settings/tokens</a>'
            "<br>И на этом же аккаунте нужно принять условия использования "
            "(кнопка «Agree and access repository») на <b>каждой</b> из моделей — "
            "speaker-diarization-3.1 использует их внутри себя:<br>"
            '1. <a href="https://huggingface.co/pyannote/speaker-diarization-3.1">'
            "pyannote/speaker-diarization-3.1</a><br>"
            '2. <a href="https://huggingface.co/pyannote/segmentation-3.0">'
            "pyannote/segmentation-3.0</a><br>"
            '3. <a href="https://huggingface.co/pyannote/speaker-diarization-community-1">'
            "pyannote/speaker-diarization-community-1</a>"
        )
        self.hint_label.setOpenExternalLinks(True)
        self.hint_label.setWordWrap(True)
        form.addRow("", self.hint_label)

        layout.addWidget(settings_box)

        # Run controls
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Запустить очередь")
        self.run_btn.clicked.connect(self.on_run_queue)
        self.stop_btn = QPushButton("Остановить после текущего файла")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop_queue)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.stop_btn)
        layout.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFixedHeight(100)
        layout.addWidget(self.log_view)

        # Results
        layout.addWidget(QLabel("Результат (выбранного файла в очереди):"))
        self.results_view = QTextEdit()
        self.results_view.setReadOnly(True)
        layout.addWidget(self.results_view, stretch=1)

        # Speaker renaming
        self.rename_group = QGroupBox("Переименовать спикеров")
        rename_layout = QVBoxLayout(self.rename_group)
        self.rename_form_widget = QWidget()
        self.rename_form = QFormLayout(self.rename_form_widget)
        rename_layout.addWidget(self.rename_form_widget)
        self.rename_apply_btn = QPushButton("Применить имена")
        self.rename_apply_btn.clicked.connect(self.on_apply_rename)
        rename_layout.addWidget(self.rename_apply_btn)
        self.rename_group.setVisible(False)
        layout.addWidget(self.rename_group)

        export_row = QHBoxLayout()
        self.export_buttons = {}
        for fmt in ("TXT", "SRT", "JSON", "DOCX"):
            btn = QPushButton(f"Экспорт {fmt}")
            btn.setEnabled(False)
            btn.clicked.connect(lambda _checked, f=fmt: self.on_export(f))
            export_row.addWidget(btn)
            self.export_buttons[fmt] = btn
        layout.addLayout(export_row)

        self.setCentralWidget(root)
        self._update_speaker_fields_visibility()
        self._update_diarization_fields_visibility(self.diarization_checkbox.isChecked())

    def _update_speaker_fields_visibility(self):
        mode = self.speakers_mode_combo.currentData()
        self.num_speakers_spin.setVisible(mode == "exact")
        self.range_row_widget.setVisible(mode == "range")

    def _update_diarization_fields_visibility(self, enabled: bool):
        for widget in (self.speakers_mode_combo, self.hf_token_edit, self.hint_label):
            self.settings_form.setRowVisible(widget, enabled)
        if enabled:
            self._update_speaker_fields_visibility()
        else:
            self.num_speakers_spin.setVisible(False)
            self.range_row_widget.setVisible(False)

    def _load_settings_into_ui(self):
        self.model_combo.setCurrentText(self.cfg.get("model_size", "large-v3"))
        self.device_combo.setCurrentText(self.cfg.get("device", "auto"))
        lang_idx = self.lang_combo.findData(self.cfg.get("language", "auto"))
        if lang_idx >= 0:
            self.lang_combo.setCurrentIndex(lang_idx)
        mode_idx = self.speakers_mode_combo.findData(self.cfg.get("speakers_mode", "auto"))
        if mode_idx >= 0:
            self.speakers_mode_combo.setCurrentIndex(mode_idx)
        self.num_speakers_spin.setValue(self.cfg.get("num_speakers", 2))
        self.min_speakers_spin.setValue(self.cfg.get("min_speakers", 1))
        self.max_speakers_spin.setValue(self.cfg.get("max_speakers", 6))
        self.hf_token_edit.setText(self.cfg.get("hf_token", ""))

    def _save_settings_from_ui(self):
        self.cfg["model_size"] = self.model_combo.currentText()
        self.cfg["device"] = self.device_combo.currentText()
        self.cfg["language"] = self.lang_combo.currentData()
        self.cfg["speakers_mode"] = self.speakers_mode_combo.currentData()
        self.cfg["num_speakers"] = self.num_speakers_spin.value()
        self.cfg["min_speakers"] = self.min_speakers_spin.value()
        self.cfg["max_speakers"] = self.max_speakers_spin.value()
        self.cfg["hf_token"] = self.hf_token_edit.text().strip()
        save_config(self.cfg)

    # ------------------------------------------------------------- queue ---
    def _refresh_queue_list(self):
        current_row = self.queue_list.currentRow()
        self.queue_list.blockSignals(True)
        self.queue_list.clear()
        for item in self.queue:
            self.queue_list.addItem(f"{Path(item.path).name}  —  {item.status}")
        self.queue_list.blockSignals(False)
        if 0 <= current_row < len(self.queue):
            self.queue_list.setCurrentRow(current_row)

    def on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите аудио или видео файлы",
            self.cfg.get("last_output_dir", ""),
            "Медиафайлы (*.mp3 *.wav *.m4a *.flac *.ogg *.mp4 *.mkv *.mov *.avi);;Все файлы (*)",
        )
        if not paths:
            return
        for path in paths:
            self.queue.append(QueueItem(path=path))
        self.cfg["last_output_dir"] = str(Path(paths[-1]).parent)
        self._refresh_queue_list()
        if self.queue_list.currentRow() < 0:
            self.queue_list.setCurrentRow(0)

    def on_remove_selected(self):
        if self.worker is not None:
            return
        row = self.queue_list.currentRow()
        if row < 0:
            return
        del self.queue[row]
        self.current_item_index = None
        self._refresh_queue_list()
        self.results_view.clear()
        self._build_rename_panel([])
        for btn in self.export_buttons.values():
            btn.setEnabled(False)

    def on_clear_queue(self):
        if self.worker is not None:
            return
        self.queue.clear()
        self.current_item_index = None
        self._refresh_queue_list()
        self.results_view.clear()
        self._build_rename_panel([])
        for btn in self.export_buttons.values():
            btn.setEnabled(False)

    def on_queue_item_selected(self, row: int):
        if row < 0 or row >= len(self.queue):
            return
        item = self.queue[row]
        if item.status == STATUS_DONE:
            self._show_item_results(row)
        elif item.status == STATUS_ERROR:
            self.current_item_index = row
            self.results_view.setPlainText(item.error)
            self._build_rename_panel([])
            for btn in self.export_buttons.values():
                btn.setEnabled(False)
        else:
            self.current_item_index = row
            self.results_view.clear()
            self._build_rename_panel([])
            for btn in self.export_buttons.values():
                btn.setEnabled(False)

    def _show_item_results(self, row: int):
        self.current_item_index = row
        chunks = self.queue[row].chunks
        self._render_results(chunks)
        self._build_rename_panel(chunks)
        enabled = bool(chunks)
        for btn in self.export_buttons.values():
            btn.setEnabled(enabled)

    # ------------------------------------------------------------ actions ---
    def on_run_queue(self):
        if self.worker is not None:
            return
        if not self.queue:
            QMessageBox.warning(self, "Очередь пуста", "Сначала добавьте хотя бы один файл.")
            return
        diarization_enabled = self.diarization_checkbox.isChecked()
        if diarization_enabled and not self.hf_token_edit.text().strip():
            QMessageBox.warning(
                self,
                "Нет токена",
                "Укажите Hugging Face токен — он нужен для загрузки модели диаризации. "
                "Либо снимите галочку «Включить диаризацию».",
            )
            return

        self._save_settings_from_ui()

        jobs: list[tuple[int, PipelineConfig]] = []
        for i, item in enumerate(self.queue):
            if item.status in (STATUS_PENDING, STATUS_ERROR):
                config = PipelineConfig(
                    input_path=item.path,
                    model_size=self.cfg["model_size"],
                    device=self.cfg["device"],
                    language=self.cfg["language"],
                    hf_token=self.cfg["hf_token"],
                    enable_diarization=diarization_enabled,
                    speakers_mode=self.cfg["speakers_mode"],
                    num_speakers=self.cfg["num_speakers"],
                    min_speakers=self.cfg["min_speakers"],
                    max_speakers=self.cfg["max_speakers"],
                )
                jobs.append((i, config))
                item.status = STATUS_PENDING
                item.error = ""

        if not jobs:
            QMessageBox.information(self, "Нечего обрабатывать", "Все файлы в очереди уже обработаны.")
            return

        self._refresh_queue_list()
        self.log_view.clear()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.add_files_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        self.worker = QueueWorker(jobs)
        self.worker.item_started.connect(self.on_item_started)
        self.worker.item_progress.connect(self.on_item_progress)
        self.worker.item_finished.connect(self.on_item_finished)
        self.worker.item_failed.connect(self.on_item_failed)
        self.worker.queue_finished.connect(self.on_queue_finished)
        self.worker.start()

    def on_stop_queue(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Остановка после текущего файла...")

    def on_item_started(self, index: int):
        self.queue[index].status = STATUS_RUNNING
        self._refresh_queue_list()
        self.queue_list.setCurrentRow(index)
        self.progress_bar.setValue(0)
        name = Path(self.queue[index].path).name
        self.status_label.setText(f"[{name}] начало обработки...")
        self.log_view.appendPlainText(f"=== {self.queue[index].path} ===")

    def on_item_progress(self, index: int, pct: int, message: str):
        self.progress_bar.setValue(pct)
        name = Path(self.queue[index].path).name
        self.status_label.setText(f"[{name}] {message}")
        self.log_view.appendPlainText(f"[{pct:3d}%] {message}")

    def on_item_finished(self, index: int, chunks: list[Chunk]):
        self.queue[index].chunks = chunks
        self.queue[index].status = STATUS_DONE
        self._refresh_queue_list()
        if self.queue_list.currentRow() == index:
            self._show_item_results(index)

    def on_item_failed(self, index: int, message: str):
        self.queue[index].status = STATUS_ERROR
        self.queue[index].error = message
        self._refresh_queue_list()
        self.log_view.appendPlainText(f"ОШИБКА [{Path(self.queue[index].path).name}]: {message}")
        if self.queue_list.currentRow() == index:
            self.results_view.setPlainText(message)

    def on_queue_finished(self):
        self.worker = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.add_files_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        done = sum(1 for i in self.queue if i.status == STATUS_DONE)
        errors = [i for i in self.queue if i.status == STATUS_ERROR]
        self.status_label.setText(f"Очередь завершена: готово {done}, ошибок {len(errors)}")
        if errors:
            names = "\n".join(Path(i.path).name for i in errors)
            QMessageBox.warning(
                self,
                "Есть ошибки",
                f"Не удалось обработать:\n{names}\n\nПодробности — в логе и в панели результата.",
            )

    def _render_results(self, chunks: list[Chunk]):
        speaker_colors: dict[str, str] = {}
        html_parts = []
        for c in chunks:
            if c.speaker not in speaker_colors:
                speaker_colors[c.speaker] = SPEAKER_COLORS[len(speaker_colors) % len(SPEAKER_COLORS)]
            color = speaker_colors[c.speaker]
            ts = f"{int(c.start // 3600):02d}:{int((c.start % 3600) // 60):02d}:{int(c.start % 60):02d}"
            html_parts.append(
                f'<p style="margin:6px 0;">'
                f'<span style="color:#6b7280;">[{ts}]</span> '
                f'<b style="color:{color};">{html.escape(c.speaker)}:</b> '
                f'{html.escape(c.text)}</p>'
            )
        self.results_view.setHtml("".join(html_parts))

    # ------------------------------------------------------ speaker rename ---
    def _build_rename_panel(self, chunks: list[Chunk]):
        while self.rename_form.rowCount():
            self.rename_form.removeRow(0)
        self.rename_edits.clear()

        speakers: list[str] = []
        for c in chunks:
            if c.speaker not in speakers:
                speakers.append(c.speaker)

        if not speakers:
            self.rename_group.setVisible(False)
            return

        self.rename_group.setVisible(True)
        for speaker in speakers:
            edit = QLineEdit(speaker)
            self.rename_edits[speaker] = edit
            self.rename_form.addRow(f"{speaker}:", edit)

    def on_apply_rename(self):
        if self.current_item_index is None:
            return
        item = self.queue[self.current_item_index]
        mapping = {}
        for old_name, edit in self.rename_edits.items():
            new_name = edit.text().strip()
            if new_name and new_name != old_name:
                mapping[old_name] = new_name
        if not mapping:
            return
        for c in item.chunks:
            if c.speaker in mapping:
                c.speaker = mapping[c.speaker]
        self._show_item_results(self.current_item_index)

    # ------------------------------------------------------------ export ---
    def on_export(self, fmt: str):
        if self.current_item_index is None:
            return
        item = self.queue[self.current_item_index]
        if not item.chunks:
            return
        ext_map = {"TXT": "txt", "SRT": "srt", "JSON": "json", "DOCX": "docx"}
        ext = ext_map[fmt]
        base_name = Path(item.path).stem
        default_name = str(Path(self.cfg.get("last_output_dir", ".")) / f"{base_name}.{ext}")
        path, _ = QFileDialog.getSaveFileName(self, f"Сохранить как {fmt}", default_name, f"*.{ext}")
        if not path:
            return
        try:
            if fmt == "TXT":
                export.to_txt(item.chunks, path)
            elif fmt == "SRT":
                export.to_srt(item.chunks, path)
            elif fmt == "JSON":
                export.to_json(item.chunks, path)
            elif fmt == "DOCX":
                export.to_docx(item.chunks, path)
            self.status_label.setText(f"Сохранено: {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))
