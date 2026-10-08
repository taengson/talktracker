"""Windows desktop GUI for TalkTracker's local sequential audio pipeline."""

from __future__ import annotations

import tkinter as tk
import subprocess
import time
from queue import Empty, Queue
from threading import Event, Thread
from pathlib import Path
from typing import TYPE_CHECKING, Any
from tkinter import filedialog, messagebox, ttk

from .core.models import WHISPER_PROFILES, WhisperProfile, get_whisper_profile

if TYPE_CHECKING:
    from .core import AnalysisResult


WINDOW_TITLE = "TalkTracker"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "windows" / "Download-Models.ps1"
PREPARE_WHISPER_SCRIPT = PROJECT_ROOT / "scripts" / "windows" / "Prepare-FasterWhisper.ps1"
INSTALL_SCRIPT = PROJECT_ROOT / "scripts" / "windows" / "Install-Dependencies.ps1"
HUGGING_FACE_CLI = PROJECT_ROOT / ".venv" / "Scripts" / "hf.exe"
SUPPORTED_FILE_TYPES = [
    ("지원되는 음성 파일", "*.m4a *.mp3 *.wav *.flac *.aac *.ogg"),
    ("모든 파일", "*.*"),
]


class TalkTrackerApp(tk.Tk):
    """Small native desktop shell used to verify the primary UI flow."""

    NAVY = "#102F38"
    TEAL = "#147D7E"
    TEAL_DARK = "#0B6263"
    MINT = "#E2F4EE"
    PAPER = "#FFFFFF"
    GROUND = "#F4F7F6"
    LINE = "#DCE7E5"
    INK = "#1E3038"
    MUTED = "#65757B"
    SUCCESS = "#28745E"
    ORANGE = "#E6A761"
    PURPLE = "#958FDD"

    def __init__(self) -> None:
        super().__init__()
        self.title(WINDOW_TITLE)
        self.geometry("1180x780")
        self.minsize(980, 680)
        self.configure(bg=self.GROUND)

        self.audio_path: Path | None = None
        self.analysis_ready = False
        self.analysis_running = False
        self.current_result: AnalysisResult | None = None
        self.preview_speaker_segments: list[Any] = []
        self.preview_duration = 0.0
        self.analysis_events: Queue[tuple[str, Any]] = Queue()
        self.cancel_event = Event()
        self.missing_models: list[str] = []
        self.model_setup_in_progress = False
        self.whisper_profile_key = tk.StringVar(value="turbo")
        self._profile_by_label = {profile.label: profile for profile in WHISPER_PROFILES}
        self.whisper_profile_label = tk.StringVar(value=get_whisper_profile("turbo").label)
        self.whisper_profile_hint = tk.StringVar(value=get_whisper_profile("turbo").description)

        self.file_name = tk.StringVar(value="아직 선택한 파일이 없습니다")
        self.file_detail = tk.StringVar(value="M4A, MP3, WAV, FLAC, AAC, OGG 파일을 선택할 수 있습니다.")
        self.status_text = tk.StringVar(value="파일을 선택하면 동작 확인을 시작할 수 있습니다.")
        self.elapsed_text = tk.StringVar(value="")
        self.progress_value = tk.DoubleVar(value=0)
        self._stage_message = self.status_text.get()
        self._stage_is_animated = False
        self._analysis_started_at: float | None = None
        self._pulse_step = 0

        self._configure_styles()
        self._build_layout()
        self.refresh_model_status()
        self._draw_timeline(empty=True)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Talk.Horizontal.TProgressbar",
            troughcolor="#E9F0EE",
            background=self.TEAL,
            bordercolor="#E9F0EE",
            lightcolor=self.TEAL,
            darkcolor=self.TEAL,
            thickness=8,
        )

    def _card(self, parent: tk.Misc, **kwargs: object) -> tk.Frame:
        options: dict[str, object] = {
            "bg": self.PAPER,
            "highlightbackground": self.LINE,
            "highlightthickness": 1,
            "bd": 0,
        }
        options.update(kwargs)
        return tk.Frame(parent, **options)

    def _label(self, parent: tk.Misc, text: str | None = None, **kwargs: object) -> tk.Label:
        options: dict[str, object] = {
            "bg": self.PAPER,
            "fg": self.INK,
            "font": ("Segoe UI", 10),
            "anchor": "w",
        }
        if text is not None:
            options["text"] = text
        options.update(kwargs)
        return tk.Label(parent, **options)

    def _button(
        self,
        parent: tk.Misc,
        text: str,
        command: object,
        *,
        primary: bool = False,
        width: int | None = None,
    ) -> tk.Button:
        background = self.TEAL if primary else self.PAPER
        foreground = self.PAPER if primary else self.NAVY
        border = self.TEAL if primary else "#C9DAD7"
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg=foreground,
            activebackground=self.TEAL_DARK if primary else "#EDF5F3",
            activeforeground=self.PAPER if primary else self.NAVY,
            disabledforeground="#8BA09C",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            padx=16,
            pady=10,
            cursor="hand2",
            width=width,
        )

    def _build_layout(self) -> None:
        viewport = tk.Frame(self, bg=self.GROUND)
        viewport.pack(fill="both", expand=True)
        viewport.grid_columnconfigure(0, weight=1)
        viewport.grid_rowconfigure(0, weight=1)

        self.content_canvas = tk.Canvas(viewport, bg=self.GROUND, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(viewport, orient="vertical", command=self.content_canvas.yview)
        self.content_canvas.configure(yscrollcommand=scrollbar.set)
        self.content_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        outer = tk.Frame(self.content_canvas, bg=self.GROUND, padx=26, pady=18)
        self.content_window = self.content_canvas.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", self._update_scroll_region)
        self.content_canvas.bind("<Configure>", self._resize_content)
        self.bind_all("<MouseWheel>", self._scroll_content, add="+")

        outer.grid_columnconfigure(0, weight=1)

        self._build_header(outer)
        self._build_hero(outer)
        self._build_input_area(outer)
        self._build_result_area(outer)

    def _update_scroll_region(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self.content_canvas.configure(scrollregion=self.content_canvas.bbox("all"))

    def _resize_content(self, event: tk.Event[tk.Misc]) -> None:
        self.content_canvas.itemconfigure(self.content_window, width=event.width)

    def _scroll_content(self, event: tk.Event[tk.Misc]) -> None:
        if event.delta:
            self.content_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _center_dialog_over_app(self, dialog: tk.Toplevel, width: int, height: int) -> None:
        """Open a helper window over TalkTracker, even on a multi-monitor desktop."""
        self.update_idletasks()
        x = self.winfo_rootx() + max(0, (self.winfo_width() - width) // 2)
        y = self.winfo_rooty() + max(0, (self.winfo_height() - height) // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")

    def _build_header(self, parent: tk.Misc) -> None:
        header = tk.Frame(parent, bg=self.GROUND)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.grid_columnconfigure(0, weight=1)

        brand = tk.Frame(header, bg=self.GROUND)
        brand.grid(row=0, column=0, sticky="w")
        tk.Label(
            brand,
            text="◔",
            bg=self.TEAL,
            fg=self.PAPER,
            font=("Segoe UI Symbol", 16, "bold"),
            width=2,
            height=1,
        ).pack(side="left", padx=(0, 9))
        tk.Label(
            brand,
            text="TalkTracker",
            bg=self.GROUND,
            fg=self.NAVY,
            font=("Segoe UI", 16, "bold"),
        ).pack(side="left")

        self.model_status_button = tk.Button(
            header,
            text="●  로컬 모델 확인 중",
            command=self.open_model_manager,
            bg="#E8F6F0",
            fg=self.SUCCESS,
            font=("Segoe UI", 9, "bold"),
            padx=11,
            pady=6,
            relief="flat",
            bd=0,
            cursor="hand2",
            activebackground="#DDF1E8",
            activeforeground=self.SUCCESS,
        )
        self.model_status_button.grid(row=0, column=1, sticky="e")

    @property
    def selected_whisper_profile(self) -> WhisperProfile:
        return get_whisper_profile(self.whisper_profile_key.get())

    def _whisper_model_label(self) -> str:
        return f"{self.selected_whisper_profile.short_label} Whisper 전사"

    def _model_locations(self) -> dict[str, tuple[Path, str]]:
        profile = self.selected_whisper_profile
        return {
            self._whisper_model_label(): (PROJECT_ROOT / "models" / profile.runtime_folder, "model.bin"),
            "Nemotron 화자 분리": (PROJECT_ROOT / "models" / "nemotron-3-diarization", "model.safetensors"),
        }

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except (ImportError, OSError):
            return False

    def refresh_model_status(self) -> None:
        """Reflect whether the selected local transcription profile is ready to load."""
        self.missing_models = [
            label
            for label, (location, weight_file) in self._model_locations().items()
            if not (location / weight_file).is_file()
        ]
        if self.missing_models:
            self.model_status_button.configure(
                text=f"●  모델 연결 필요 ({len(self.missing_models)})",
                bg="#FFF1E4",
                fg="#99601C",
                activebackground="#FFE5CA",
                activeforeground="#99601C",
            )
            return

        self.model_status_button.configure(
            text="●  로컬 모델 연결됨",
            bg="#E8F6F0",
            fg=self.SUCCESS,
            activebackground="#DDF1E8",
            activeforeground=self.SUCCESS,
        )

    def open_model_manager(self) -> None:
        """Show local model state and offer the existing setup scripts."""
        self.refresh_model_status()
        dialog = tk.Toplevel(self)
        dialog.title("로컬 모델 관리")
        dialog.resizable(False, False)
        dialog.configure(bg=self.GROUND)
        dialog.transient(self)
        self._center_dialog_over_app(dialog, 500, 390)

        header = tk.Frame(dialog, bg=self.GROUND)
        header.pack(fill="x", padx=24, pady=(22, 10))
        tk.Label(header, text="로컬 모델 관리", bg=self.GROUND, fg=self.NAVY, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        explanation = (
            "TalkTracker는 프로젝트 안의 models 폴더에서만 모델을 찾습니다.\n"
            "모델이 없으면 아래 버튼으로 Windows 다운로드 스크립트를 실행할 수 있습니다."
        )
        tk.Label(header, text=explanation, bg=self.GROUND, fg=self.MUTED, font=("Segoe UI", 9), justify="left").pack(anchor="w", pady=(5, 0))

        state_card = self._card(dialog, padx=16, pady=11)
        state_card.pack(fill="x", padx=24, pady=(0, 13))
        for label, (location, weight_file) in self._model_locations().items():
            ready = label not in self.missing_models
            row = tk.Frame(state_card, bg=self.PAPER)
            row.pack(fill="x", pady=5)
            symbol = "●" if ready else "!"
            color = self.SUCCESS if ready else "#A8631A"
            description = "준비됨" if ready else "다운로드 필요"
            tk.Label(row, text=symbol, bg=self.PAPER, fg=color, font=("Segoe UI", 10, "bold"), width=2).pack(side="left")
            tk.Label(row, text=label, bg=self.PAPER, fg=self.NAVY, font=("Segoe UI", 10, "bold"), anchor="w").pack(side="left")
            tk.Label(row, text=description, bg=self.PAPER, fg=color, font=("Segoe UI", 9, "bold"), anchor="e").pack(side="right")

        info = tk.Label(
            dialog,
            text=f"저장 위치: {PROJECT_ROOT / 'models'}",
            bg=self.GROUND,
            fg=self.MUTED,
            font=("Segoe UI", 8),
            anchor="w",
        )
        info.pack(fill="x", padx=24)

        actions = tk.Frame(dialog, bg=self.GROUND)
        actions.pack(fill="x", padx=24, pady=(18, 24))
        self._button(actions, "상태 새로고침", lambda: self._reload_model_manager(dialog), width=12).pack(side="left")

        if self.missing_models:
            profile = self.selected_whisper_profile
            source_model_dir = PROJECT_ROOT / "models" / profile.source_folder
            source_models_missing = not (source_model_dir / "model.safetensors").is_file()
            nemotron_missing = "Nemotron 화자 분리" in self.missing_models
            whisper_runtime_missing = self._whisper_model_label() in self.missing_models
            if not HUGGING_FACE_CLI.is_file():
                self._button(
                    actions,
                    "패키지 준비 시작",
                    lambda: self.start_package_install(dialog),
                    primary=True,
                ).pack(side="right")
            elif profile.requires_gpu and not self._gpu_available():
                tk.Label(
                    actions,
                    text="Large-v3는 호환 GPU가 있는 PC에서 준비할 수 있습니다.",
                    bg=self.GROUND,
                    fg="#99601C",
                    font=("Segoe UI", 9, "bold"),
                ).pack(side="right")
            elif source_models_missing or nemotron_missing:
                self._button(
                    actions,
                    "모델 다운로드 시작",
                    lambda: self.start_model_download(dialog),
                    primary=True,
                ).pack(side="right")
            elif whisper_runtime_missing:
                self._button(
                    actions,
                    "선택 모델 준비",
                    lambda: self.start_whisper_prepare(dialog),
                    primary=True,
                ).pack(side="right")
        else:
            self._button(actions, "닫기", dialog.destroy, width=8).pack(side="right")

    def _reload_model_manager(self, dialog: tk.Toplevel) -> None:
        self.model_setup_in_progress = False
        dialog.destroy()
        self.refresh_model_status()
        self.open_model_manager()

    def _start_windows_script(self, script: Path, description: str, arguments: list[str] | None = None) -> bool:
        if self.model_setup_in_progress:
            messagebox.showinfo(WINDOW_TITLE, "모델 준비 작업이 이미 실행 중입니다.\n완료된 뒤 ‘상태 새로고침’을 눌러 주세요.")
            return False
        if not script.is_file():
            messagebox.showerror(WINDOW_TITLE, f"실행 스크립트를 찾지 못했습니다.\n\n{script}")
            return False
        try:
            command = ["powershell.exe", "-NoExit", "-ExecutionPolicy", "Bypass", "-File", str(script)]
            if arguments:
                command.extend(arguments)
            subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
            )
        except OSError as error:
            messagebox.showerror(WINDOW_TITLE, f"스크립트를 시작하지 못했습니다.\n\n{error}")
            return False

        self.model_setup_in_progress = True
        messagebox.showinfo(
            WINDOW_TITLE,
            f"{description} 창을 열었습니다.\n\n"
            "별도로 열린 창에 Done 메시지가 나올 때까지 기다려 주세요.\n"
            "이 모델 관리 창은 닫지 말고, 완료된 뒤 ‘상태 새로고침’을 눌러 주세요.",
        )
        return True

    def start_model_download(self, dialog: tk.Toplevel) -> None:
        profile = self.selected_whisper_profile
        nemotron_weights = PROJECT_ROOT / "models" / "nemotron-3-diarization" / "model.safetensors"
        download_message = f"{profile.label} 모델을 다운로드한 뒤, TalkTracker 실행용으로 준비합니다."
        if not nemotron_weights.is_file():
            download_message += "\nNemotron 화자 분리 모델도 함께 다운로드합니다."
        if not messagebox.askyesno(
            WINDOW_TITLE,
            f"{download_message}\n저장 공간과 인터넷 연결이 필요하며, 몇 분 정도 걸릴 수 있습니다.\n\n계속할까요?",
        ):
            return
        arguments = ["-WhisperProfile", profile.key, "-PrepareForLocalUse"]
        if nemotron_weights.is_file():
            arguments.append("-SkipDiarization")
        self._start_windows_script(DOWNLOAD_SCRIPT, "모델 다운로드 및 준비", arguments)

    def start_package_install(self, dialog: tk.Toplevel) -> None:
        self._start_windows_script(INSTALL_SCRIPT, "앱 패키지 준비")

    def start_whisper_prepare(self, dialog: tk.Toplevel) -> None:
        profile = self.selected_whisper_profile
        if not messagebox.askyesno(
            WINDOW_TITLE,
            f"{profile.label} 모델을 로컬 실행용 형식으로 준비합니다.\n추가 저장 공간이 필요하며, 몇 분 정도 걸릴 수 있습니다.\n\n계속할까요?",
        ):
            return
        self._start_windows_script(PREPARE_WHISPER_SCRIPT, "선택 모델 준비", ["-WhisperProfile", profile.key])

    def _build_hero(self, parent: tk.Misc) -> None:
        hero = tk.Frame(parent, bg=self.NAVY, padx=32, pady=25)
        hero.grid(row=1, column=0, sticky="ew", pady=(0, 18))
        hero.grid_columnconfigure(0, weight=1)

        tk.Label(
            hero,
            text="LOCAL SPEAKER DIARIZATION",
            bg=self.NAVY,
            fg="#9EE1D3",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            hero,
            text="대화의 흐름을 한눈에 정리하세요.",
            bg=self.NAVY,
            fg=self.PAPER,
            font=("Segoe UI", 25, "bold"),
        ).grid(row=1, column=0, sticky="w", pady=(7, 6))
        tk.Label(
            hero,
            text="음성 파일을 선택하면 TalkTracker가 화자 구간과 대화 내용을 순서대로 정리합니다.\n파일과 모델 처리는 이 PC 안에서만 이뤄집니다.",
            bg=self.NAVY,
            fg="#CAE5DF",
            font=("Segoe UI", 10),
            justify="left",
        ).grid(row=2, column=0, sticky="w")

    def _build_input_area(self, parent: tk.Misc) -> None:
        content = tk.Frame(parent, bg=self.GROUND)
        content.grid(row=2, column=0, sticky="ew", pady=(0, 20))
        content.grid_columnconfigure(0, weight=7)
        content.grid_columnconfigure(1, weight=3)

        select_card = self._card(content, padx=22, pady=20)
        select_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._label(select_card, "새 녹음 정리", font=("Segoe UI", 14, "bold"), fg=self.NAVY).pack(anchor="w")
        self._label(
            select_card,
            "파일 하나만 선택하세요. 나머지 흐름은 TalkTracker가 안내합니다.",
            fg=self.MUTED,
        ).pack(anchor="w", pady=(4, 16))

        profile_row = tk.Frame(select_card, bg=self.PAPER)
        profile_row.pack(fill="x", pady=(0, 12))
        self._label(profile_row, "전사 품질", font=("Segoe UI", 9, "bold"), fg=self.NAVY).grid(row=0, column=0, sticky="w")
        self.whisper_selector = ttk.Combobox(
            profile_row,
            textvariable=self.whisper_profile_label,
            values=[profile.label for profile in WHISPER_PROFILES],
            state="readonly",
            width=34,
            font=("Segoe UI", 9),
        )
        self.whisper_selector.grid(row=0, column=1, sticky="e")
        self.whisper_selector.bind("<<ComboboxSelected>>", self._on_whisper_profile_changed)
        self._label(profile_row, textvariable=self.whisper_profile_hint, fg=self.MUTED, font=("Segoe UI", 8)).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(4, 0),
        )
        profile_row.grid_columnconfigure(0, weight=1)

        chooser = tk.Frame(select_card, bg="#F3FAF8", highlightbackground="#A7C8C1", highlightthickness=1, padx=17, pady=15)
        chooser.pack(fill="x")
        chooser.grid_columnconfigure(0, weight=1)
        self._label(chooser, textvariable=self.file_name, bg="#F3FAF8", font=("Segoe UI", 10, "bold"), fg=self.NAVY).grid(row=0, column=0, sticky="w")
        self._label(chooser, textvariable=self.file_detail, bg="#F3FAF8", fg=self.MUTED, wraplength=440).grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.choose_button = self._button(chooser, "파일 선택", self.choose_file, width=10)
        self.choose_button.grid(row=0, column=1, rowspan=2, padx=(16, 0))

        actions = tk.Frame(select_card, bg=self.PAPER)
        actions.pack(fill="x", pady=(14, 0))
        self.start_button = self._button(actions, "분석 시작", self.start_analysis, primary=True)
        self.start_button.pack(side="left")
        self.clear_button = self._button(actions, "선택 취소", self.clear_file)
        self.clear_button.pack(side="left", padx=(8, 0))
        self.cancel_button = self._button(actions, "중단", self.cancel_analysis)
        self.cancel_button.pack(side="left", padx=(8, 0))
        self.cancel_button.configure(state="disabled", cursor="arrow")

        guide_card = self._card(content, padx=20, pady=18)
        guide_card.grid(row=0, column=1, sticky="nsew")
        self._label(guide_card, "사용 방법", font=("Segoe UI", 13, "bold"), fg=self.NAVY).pack(anchor="w", pady=(0, 10))
        for number, title, detail in (
            ("1", "녹음 파일 선택", "아이폰 음성 메모도 바로 넣을 수 있어요."),
            ("2", "분석 시작", "화자 분리와 전사를 순서대로 처리합니다."),
            ("3", "결과 저장", "대화 기록을 Markdown 문서로 저장합니다."),
        ):
            row = tk.Frame(guide_card, bg=self.PAPER)
            row.pack(fill="x", pady=5)
            tk.Label(row, text=number, bg=self.MINT, fg=self.TEAL_DARK, font=("Segoe UI", 9, "bold"), width=3, pady=4).pack(side="left", padx=(0, 8))
            text = tk.Frame(row, bg=self.PAPER)
            text.pack(side="left", fill="x", expand=True)
            self._label(text, title, font=("Segoe UI", 9, "bold"), fg=self.NAVY).pack(anchor="w")
            self._label(text, detail, font=("Segoe UI", 8), fg=self.MUTED, wraplength=205).pack(anchor="w")

    def _on_whisper_profile_changed(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        profile = self._profile_by_label.get(self.whisper_profile_label.get())
        if not profile:
            return
        self.whisper_profile_key.set(profile.key)
        self.whisper_profile_hint.set(profile.description)
        self.refresh_model_status()
        if profile.requires_gpu and not self._gpu_available():
            self._set_status("최고 정확도 모델은 호환 GPU가 필요합니다. 현재 PC에서는 실행할 수 없습니다.")
        elif self.missing_models:
            self._set_status(f"{profile.short_label} 전사 모델을 준비한 뒤 분석할 수 있습니다.")
        else:
            self._set_status(f"{profile.short_label} 전사 모델을 사용할 준비가 됐습니다.")

    def _build_result_area(self, parent: tk.Misc) -> None:
        card = self._card(parent, padx=22, pady=20)
        card.grid(row=3, column=0, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(4, weight=1)

        heading = tk.Frame(card, bg=self.PAPER)
        heading.grid(row=0, column=0, sticky="ew")
        heading.grid_columnconfigure(0, weight=1)
        self._label(heading, "대화 기록", font=("Segoe UI", 15, "bold"), fg=self.NAVY).grid(row=0, column=0, sticky="w")
        self._button(heading, "결과 저장", self.save_result, width=10).grid(row=0, column=1, sticky="e")

        status = tk.Frame(card, bg="#EDF9F5", padx=13, pady=9)
        status.grid(row=1, column=0, sticky="ew", pady=(12, 8))
        tk.Label(status, text="●", bg="#EDF9F5", fg="#50AE8C", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 7))
        tk.Label(status, textvariable=self.elapsed_text, bg="#EDF9F5", fg="#47746A", font=("Consolas", 9, "bold"), anchor="e").pack(side="right")
        tk.Label(status, textvariable=self.status_text, bg="#EDF9F5", fg=self.SUCCESS, font=("Segoe UI", 9, "bold"), anchor="w").pack(side="left", fill="x", expand=True)

        self.progress = ttk.Progressbar(card, variable=self.progress_value, maximum=100, style="Talk.Horizontal.TProgressbar")
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 14))

        self.timeline_canvas = tk.Canvas(card, height=94, bg="#F9FBFA", highlightthickness=1, highlightbackground=self.LINE)
        self.timeline_canvas.grid(row=3, column=0, sticky="ew", pady=(0, 13))
        self.timeline_canvas.bind("<Configure>", lambda _event: self._draw_timeline(empty=not self._has_timeline()))

        self.transcript = tk.Text(
            card,
            height=8,
            wrap="word",
            bg=self.PAPER,
            fg=self.INK,
            font=("Segoe UI", 10),
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.LINE,
            padx=15,
            pady=12,
            state="disabled",
        )
        self.transcript.grid(row=4, column=0, sticky="nsew")
        self._set_transcript(
            "음성 파일을 선택한 뒤 ‘분석 시작’을 누르면 결과가 표시됩니다.\n\n"
            "처리 순서: 음성 정규화 → Nemotron 화자 분리 → Whisper 전사 → 시간 기준 결합"
        )

    def choose_file(self) -> None:
        selected = filedialog.askopenfilename(title="음성 파일 선택", filetypes=SUPPORTED_FILE_TYPES)
        if not selected:
            return

        self.audio_path = Path(selected)
        try:
            size_mb = self.audio_path.stat().st_size / (1024 * 1024)
            detail = f"{self.audio_path.suffix.removeprefix('.').upper()} · {size_mb:.1f} MB · 로컬 처리 준비 완료"
        except OSError:
            detail = "파일 정보를 읽지 못했지만, 선택 상태는 유지합니다."

        self.file_name.set(self.audio_path.name)
        self.file_detail.set(detail)
        self.analysis_ready = False
        self.current_result = None
        self.preview_speaker_segments = []
        self.preview_duration = 0.0
        self.progress_value.set(0)
        self.elapsed_text.set("")
        self._set_status("파일을 선택했습니다. ‘분석 시작’을 누르면 로컬 분석을 시작합니다.")
        self._draw_timeline(empty=True)

    def clear_file(self) -> None:
        self.audio_path = None
        self.analysis_ready = False
        self.current_result = None
        self.preview_speaker_segments = []
        self.preview_duration = 0.0
        self.progress_value.set(0)
        self.elapsed_text.set("")
        self.file_name.set("아직 선택한 파일이 없습니다")
        self.file_detail.set("M4A, MP3, WAV, FLAC, AAC, OGG 파일을 선택할 수 있습니다.")
        self._set_status("파일을 선택하면 동작 확인을 시작할 수 있습니다.")
        self._set_transcript(
            "음성 파일을 선택한 뒤 ‘분석 시작’을 누르면 결과가 표시됩니다.\n\n"
            "처리 순서: 음성 정규화 → Nemotron 화자 분리 → Whisper 전사 → 시간 기준 결합"
        )
        self._draw_timeline(empty=True)

    def start_analysis(self) -> None:
        if not self.audio_path:
            messagebox.showinfo(WINDOW_TITLE, "먼저 확인할 음성 파일을 선택해 주세요.")
            return
        profile = self.selected_whisper_profile
        if profile.requires_gpu and not self._gpu_available():
            messagebox.showwarning(
                WINDOW_TITLE,
                "‘최고 정확도’ 모델은 호환 CUDA GPU가 필요합니다.\n\n"
                "현재 PC에서는 ‘권장’ 또는 ‘빠른 전사’를 선택해 주세요.",
            )
            return
        self.refresh_model_status()
        if self.missing_models:
            messagebox.showwarning(WINDOW_TITLE, "로컬 모델을 준비한 뒤 분석할 수 있습니다.")
            self.open_model_manager()
            return

        self.analysis_ready = False
        self.analysis_running = True
        self.current_result = None
        self.preview_speaker_segments = []
        self.preview_duration = 0.0
        self.analysis_events = Queue()
        self.cancel_event = Event()
        self.progress_value.set(0)
        self.start_button.configure(state="disabled", cursor="arrow")
        self.choose_button.configure(state="disabled", cursor="arrow")
        self.clear_button.configure(state="disabled", cursor="arrow")
        self.cancel_button.configure(state="normal", cursor="hand2")
        self.whisper_selector.configure(state="disabled")
        self._analysis_started_at = time.monotonic()
        self._pulse_step = 0
        self._set_status("로컬 분석을 시작하고 있습니다", animate=True)
        self._draw_timeline(empty=True)
        self._update_activity_indicator()
        source = self.audio_path
        Thread(target=self._analysis_worker, args=(source, profile.key), daemon=True).start()
        self.after(100, self._poll_analysis_events)

    def _analysis_worker(self, source: Path, whisper_profile: str) -> None:
        try:
            from .core import AnalysisEngine

            engine = AnalysisEngine(PROJECT_ROOT, whisper_profile=whisper_profile)
            result = engine.analyze(
                source,
                progress=self._queue_progress,
                on_diarization=self._queue_diarization,
                should_cancel=self.cancel_event.is_set,
            )
        except Exception as error:  # Converted to a readable dialog on the GUI thread.
            self.analysis_events.put(("error", error))
            return
        self.analysis_events.put(("partial" if result.is_partial else "result", result))

    def _queue_progress(self, value: int, message: str) -> None:
        self.analysis_events.put(("progress", (value, message)))

    def _queue_diarization(self, segments: list[Any], duration: float) -> None:
        self.analysis_events.put(("diarization", (segments, duration)))

    def _poll_analysis_events(self) -> None:
        while True:
            try:
                event, payload = self.analysis_events.get_nowait()
            except Empty:
                break
            if event == "progress":
                value, message = payload
                self.progress_value.set(value)
                self._set_status(message, animate=True)
            elif event == "diarization":
                self.preview_speaker_segments, self.preview_duration = payload
                self._draw_timeline(empty=False)
            elif event == "result":
                self._analysis_succeeded(payload)
                return
            elif event == "partial":
                self._analysis_cancelled(payload)
                return
            elif event == "error":
                self._analysis_failed(payload)
                return
        if self.analysis_running:
            self.after(100, self._poll_analysis_events)

    def _analysis_succeeded(self, result: AnalysisResult) -> None:
        elapsed = self._elapsed_seconds()
        self.analysis_running = False
        self.analysis_ready = True
        self.current_result = result
        self.preview_speaker_segments = []
        self.preview_duration = 0.0
        self.progress_value.set(100)
        self.elapsed_text.set(f"총 {_format_elapsed(elapsed)}")
        self._set_status(f"분석 완료 · 화자 {len(result.speakers)}명의 대화 기록을 만들었습니다.")
        self.start_button.configure(state="normal", cursor="hand2")
        self.choose_button.configure(state="normal", cursor="hand2")
        self.clear_button.configure(state="normal", cursor="hand2")
        self.cancel_button.configure(state="disabled", cursor="arrow")
        self.whisper_selector.configure(state="readonly")
        self._draw_timeline(empty=False)
        self._show_analysis_result(result)

    def _analysis_failed(self, error: Exception) -> None:
        elapsed = self._elapsed_seconds()
        self.analysis_running = False
        self.progress_value.set(0)
        self.elapsed_text.set(f"총 {_format_elapsed(elapsed)}")
        self._set_status("분석을 완료하지 못했습니다.")
        self.start_button.configure(state="normal", cursor="hand2")
        self.choose_button.configure(state="normal", cursor="hand2")
        self.clear_button.configure(state="normal", cursor="hand2")
        self.cancel_button.configure(state="disabled", cursor="arrow")
        self.whisper_selector.configure(state="readonly")
        messagebox.showerror(WINDOW_TITLE, f"음성 파일을 처리하지 못했습니다.\n\n{error}")

    def cancel_analysis(self) -> None:
        if not self.analysis_running:
            return
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled", cursor="arrow")
        self._set_status("중단 요청을 처리하고 있습니다", animate=True)

    def _analysis_cancelled(self, result: AnalysisResult) -> None:
        elapsed = self._elapsed_seconds()
        self.analysis_running = False
        self.analysis_ready = bool(result.speaker_segments or result.utterances)
        self.current_result = result
        self.preview_speaker_segments = []
        self.preview_duration = 0.0
        self.elapsed_text.set(f"총 {_format_elapsed(elapsed)}")
        if result.utterances:
            message = "중단됨 · 전사된 부분까지의 대화 기록을 표시했습니다."
        elif result.speaker_segments:
            message = "중단됨 · 완료된 화자 타임라인을 표시했습니다."
        else:
            message = "중단됨 · 완료된 분석 결과가 아직 없습니다."
        self._set_status(message)
        self.start_button.configure(state="normal", cursor="hand2")
        self.choose_button.configure(state="normal", cursor="hand2")
        self.clear_button.configure(state="normal", cursor="hand2")
        self.cancel_button.configure(state="disabled", cursor="arrow")
        self.whisper_selector.configure(state="readonly")
        self._draw_timeline(empty=not bool(result.speaker_segments))
        self._show_analysis_result(result)

    def _draw_timeline(self, *, empty: bool) -> None:
        canvas = self.timeline_canvas
        canvas.delete("all")
        result = self.current_result
        segments = result.speaker_segments if result else self.preview_speaker_segments
        duration = result.duration if result else self.preview_duration
        if empty or not segments:
            canvas.configure(height=94)
            canvas.create_text(18, 18, text="화자 타임라인", fill=self.NAVY, anchor="w", font=("Segoe UI", 10, "bold"))
            canvas.create_text(18, 55, text="분석을 시작하면 화자별 발화 구간이 이곳에 표시됩니다.", fill=self.MUTED, anchor="w", font=("Segoe UI", 9))
            return

        width = max(canvas.winfo_width(), 700)
        speakers = list(dict.fromkeys(segment.speaker for segment in segments))
        colors = ("#66B9AA", self.ORANGE, self.PURPLE, "#6BAAC0", "#D68799", "#9CB568", "#B293CE", "#C59D67")
        canvas.configure(height=max(94, 38 + len(speakers) * 22))
        canvas.create_text(18, 13, text="화자 타임라인", fill=self.NAVY, anchor="w", font=("Segoe UI", 9, "bold"))
        track_x, track_width = 86, width - 110
        duration = max(duration, 0.1)
        row_index = {speaker: index for index, speaker in enumerate(speakers)}
        for speaker, index in row_index.items():
            y = 34 + index * 18
            canvas.create_text(18, y + 5, text=speaker, fill=self.MUTED, anchor="w", font=("Segoe UI", 8, "bold"))
            canvas.create_rectangle(track_x, y, track_x + track_width, y + 10, fill="#EAF0EF", outline="")
        for segment in segments:
            index = row_index[segment.speaker]
            y = 34 + index * 18
            color = colors[index % len(colors)]
            start = track_x + track_width * (segment.start / duration)
            end = track_x + track_width * (min(segment.end, duration) / duration)
            canvas.create_rectangle(start, y, max(start + 2, end), y + 10, fill=color, outline="")

    def _has_timeline(self) -> bool:
        return bool(self.current_result or self.preview_speaker_segments)

    def _set_status(self, message: str, *, animate: bool = False) -> None:
        self._stage_message = message.rstrip(".… ")
        self._stage_is_animated = animate and self.analysis_running
        if self._stage_is_animated:
            self.status_text.set(f"{self._stage_message}{'.' * (self._pulse_step % 3 + 1)}")
        else:
            self.status_text.set(message)

    def _update_activity_indicator(self) -> None:
        if not self.analysis_running:
            return
        self.elapsed_text.set(f"경과 {_format_elapsed(self._elapsed_seconds())}")
        if self._stage_is_animated:
            self.status_text.set(f"{self._stage_message}{'.' * (self._pulse_step % 3 + 1)}")
            self._pulse_step = (self._pulse_step + 1) % 3
        self.after(1_000, self._update_activity_indicator)

    def _elapsed_seconds(self) -> int:
        if self._analysis_started_at is None:
            return 0
        return int(time.monotonic() - self._analysis_started_at)

    def _show_analysis_result(self, result: AnalysisResult) -> None:
        completion_label = "중단 시점까지의 결과" if result.is_partial else "분석 완료"
        lines = [f"{result.source_name} · 화자 {len(result.speakers)}명 · {_format_time(result.duration)} · {completion_label}", ""]
        if not result.utterances:
            if result.is_partial:
                lines.append("중단 시점까지 완료된 전사문이 없습니다. 화자 타임라인은 그대로 확인할 수 있습니다.")
            else:
                lines.append("전사문을 찾지 못했습니다. 음질 또는 녹음 내용을 확인해 주세요.")
        for utterance in result.utterances:
            lines.extend(
                [
                    f"{utterance.speaker}  |  {_format_time(utterance.start)} – {_format_time(utterance.end)}",
                    utterance.text,
                    "",
                ]
            )
        text = "\n".join(lines).rstrip()
        self._set_transcript(text)

    def _set_transcript(self, text: str) -> None:
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.insert("1.0", text)
        self.transcript.configure(state="disabled")

    def save_result(self) -> None:
        if not self.analysis_ready:
            messagebox.showinfo(WINDOW_TITLE, "분석이 끝난 뒤 대화 기록을 저장할 수 있습니다.")
            return

        source_name = self.audio_path.stem if self.audio_path else "talktracker-result"
        target = filedialog.asksaveasfilename(
            title="대화 기록 저장",
            initialfile=f"{source_name}-talktracker.md",
            defaultextension=".md",
            filetypes=[("Markdown 문서", "*.md")],
        )
        if not target:
            return

        if not self.current_result:
            messagebox.showerror(WINDOW_TITLE, "저장할 분석 결과를 찾지 못했습니다.")
            return
        content = self.current_result.to_markdown()
        try:
            Path(target).write_text(content, encoding="utf-8")
        except OSError as error:
            messagebox.showerror(WINDOW_TITLE, f"결과를 저장하지 못했습니다.\n\n{error}")
            return
        messagebox.showinfo(WINDOW_TITLE, f"결과를 저장했습니다.\n\n{target}")

def run() -> None:
    app = TalkTrackerApp()
    app.mainloop()


def _format_time(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    return f"{minutes:02}:{seconds:02}"


def _format_elapsed(seconds: int) -> str:
    hours, remainder = divmod(max(0, seconds), 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"
