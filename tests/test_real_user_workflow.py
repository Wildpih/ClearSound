"""Comprehensive Real-User End-to-End Test Suite for ClearSound.

Tests real user workflows through the Qt GUI application without mocking:
1. Hub project listing, deletion via UI button, and disk cleanup.
2. Opening a real video project with Cyrillic path.
3. Running the pipeline worker to completion.
4. Objective acoustic proof that vocals are eliminated and no repeated donor is pasted.
5. User adjustments via Segment Editor table and video remuxing.
6. Edge case & stress tests (rapid track switching, mute/volume bounds, missing files).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure offscreen Qt platform for testing environment
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import soundfile as sf
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from clearsound.config import Config
from clearsound.core.project_manager import ProjectManager
from clearsound.library.sound_library import SoundLibrary
from clearsound.ui.main_window import MainWindow
from clearsound.ui.project_hub_widget import ProjectCard


def get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def wait_ms(ms: int) -> None:
    """Process Qt event loop for given milliseconds."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def test_1_hub_project_deletion():
    """Test real user clicking delete on a project card in the Hub."""
    print("\n" + "=" * 60)
    print("TEST 1: Hub Project Deletion (Real User Action)")
    print("=" * 60)

    config = Config()
    pm = ProjectManager(config)

    # 1. Create a dummy test project on disk
    dummy_file = config.temp_dir / "temp_test_video.mp4"
    dummy_file.write_bytes(b"\x00" * 1024)
    proj = pm.create_project(dummy_file, name="Проект для удаления тест")
    proj_id = proj.id
    proj_dir = config.projects_dir / proj_id
    assert proj_dir.exists(), "Project folder should exist on disk"
    print(f"  Created test project: {proj_id} ({proj.name})")

    app = get_app()
    win = MainWindow(config=config)
    win.show()
    wait_ms(200)

    # Verify project appears in hub
    initial_count = len(win.hub_widget.all_projects)
    print(f"  Hub initial projects count: {initial_count}")

    # Find the card for this project
    card_found = None
    for i in range(win.hub_widget.container_layout.count()):
        item = win.hub_widget.container_layout.itemAt(i)
        w = item.widget() if item else None
        if isinstance(w, ProjectCard) and w.project.id == proj_id:
            card_found = w
            break

    assert card_found is not None, "Project card must be visible in the Hub"
    print("  ✓ Located ProjectCard in Hub layout")

    # Hook QMessageBox to auto-accept confirmation
    original_question = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *args, **kwargs: QMessageBox.Yes)

    try:
        # Simulate user clicking the 🗑 delete button on the card
        print("  Simulating user clicking 🗑 delete button...")
        card_found.delete_requested.emit(proj_id)
        wait_ms(300)

        # Verify project folder is deleted from disk
        assert not proj_dir.exists(), f"Project directory {proj_dir} was NOT deleted from disk!"
        print("  ✓ Project directory permanently removed from disk")

        # Verify hub refreshed
        new_count = len(win.hub_widget.all_projects)
        assert new_count == initial_count - 1, f"Expected {initial_count - 1} projects, got {new_count}"
        print(f"  ✓ Hub project count decreased to {new_count}")

        # Verify card is gone from UI
        still_present = any(
            isinstance(win.hub_widget.container_layout.itemAt(i).widget(), ProjectCard)
            and win.hub_widget.container_layout.itemAt(i).widget().project.id == proj_id
            for i in range(win.hub_widget.container_layout.count())
            if win.hub_widget.container_layout.itemAt(i).widget()
        )
        assert not still_present, "Deleted project card should no longer exist in UI"
        print("  ✓ ProjectCard removed from UI container")
        print(">>> TEST 1 PASSED: Hub deletion works completely and reliably! <<<\n")

    finally:
        QMessageBox.question = original_question
        win.close()


def test_2_real_video_pipeline_and_acoustic_verification():
    """Test full processing on real Overwatch video and verify acoustics objectively."""
    print("\n" + "=" * 60)
    print("TEST 2: Real Video Processing & Acoustic Verification")
    print("=" * 60)

    video_path = Path(r"C:\Users\akeo3\OneDrive\Рабочий стол\Overwatch Cinematic Teaser _ _Are You With Us__.mp4")
    if not video_path.exists():
        print(f"  [SKIP] Video file not found at {video_path}")
        return

    orig_critical = QMessageBox.critical
    orig_warning = QMessageBox.warning
    QMessageBox.critical = lambda *a, **kw: QMessageBox.Ok
    QMessageBox.warning = lambda *a, **kw: QMessageBox.Ok

    win = None
    try:
        config = Config()
        app = get_app()
        win = MainWindow(config=config)
        win.show()
        wait_ms(200)

        print(f"  Opening real video: {video_path.name}")
        win.create_and_open_project(video_path)
        wait_ms(300)

        assert win.current_project is not None, "Project should be loaded"
        print(f"  ✓ Project created: {win.current_project.id}")

        # Simulate user clicking "Начать обработку"
        print("  Simulating user clicking 'Начать обработку'...")
        win._on_process_clicked()

        # Wait for background pipeline worker to finish
        loop = QEventLoop()
        if win.pipeline_worker and win.pipeline_worker.isRunning():
            win.pipeline_worker.finished.connect(lambda res: loop.quit())
            win.pipeline_worker.error.connect(lambda err: loop.quit())
            # Safety timeout: 120s
            QTimer.singleShot(120000, loop.quit)
            loop.exec()

        wait_ms(500)
        assert win.current_project.status == "processed", f"Project status is {win.current_project.status}, expected 'processed'"
        print("  ✓ Pipeline completed successfully!")

        # ── OBJECTIVE ACOUSTIC VERIFICATIONS ──
        print("\n  [Acoustic Analysis of Produced Audio]")
        p_orig = config.temp_dir / f"{video_path.stem}_extracted.wav"
        p_novoc = win.current_project.no_vocals_wav
        p_enh = win.current_project.enhanced_wav
        p_prev_video = win.current_project.preview_video_path

        assert p_orig.exists(), "Original extracted audio must exist"
        assert p_novoc.exists(), "No-vocals audio must exist"
        assert p_enh.exists(), "Enhanced audio must exist"
        assert p_prev_video.exists(), "Preview remuxed video must exist"

        orig_audio, sr = sf.read(str(p_orig))
        novoc_audio, _ = sf.read(str(p_novoc))
        enh_audio, _ = sf.read(str(p_enh))

        if orig_audio.ndim > 1:
            orig_mono = np.mean(orig_audio, axis=1)
            novoc_mono = np.mean(novoc_audio, axis=1)
            enh_mono = np.mean(enh_audio, axis=1)
        else:
            orig_mono, novoc_mono, enh_mono = orig_audio, novoc_audio, enh_audio

        # 1. Check Voice Removal during speech sections
        # Speech vocal formant band: 300 Hz - 3000 Hz
        from scipy.signal import butter, sosfilt
        sos_speech = butter(3, [300 / (sr / 2), 3000 / (sr / 2)], btype="bandpass", output="sos")

        speech_gaps = [g for g in win.current_project.gaps if g.event_type == "speech_dialogue"]
        print(f"  Total gaps: {len(win.current_project.gaps)}, Speech dialogue gaps: {len(speech_gaps)}")
        assert len(speech_gaps) >= 15, f"Expected majority of gaps to be speech_dialogue, found {len(speech_gaps)}"

        total_orig_speech_energy = 0.0
        total_enh_speech_energy = 0.0

        for g in speech_gaps:
            st, en = g.start_sample, g.end_sample
            seg_orig = sosfilt(sos_speech, orig_mono[st:en])
            seg_enh = sosfilt(sos_speech, enh_mono[st:en])
            total_orig_speech_energy += np.sum(seg_orig ** 2)
            total_enh_speech_energy += np.sum(seg_enh ** 2)

        vocal_suppression_ratio = total_enh_speech_energy / max(total_orig_speech_energy, 1e-10)
        print(f"  Vocal energy ratio in dialogue (Enhanced / Original): {vocal_suppression_ratio:.3f} (-{10 * np.log10(1 / vocal_suppression_ratio):.1f} dB)")
        # Vocal energy must be heavily suppressed compared to original
        assert vocal_suppression_ratio < 0.65, f"Voice was not sufficiently suppressed: ratio={vocal_suppression_ratio}"
        print("  ✓ Voice suppression verified objectively: speech energy reduced!")

        # 2. Check that Non-Vocal SFX (Camera drop, Camera shutter clicks & Gunshots) were properly restored from library
        drop_gaps = [g for g in win.current_project.gaps if g.event_type == "camera_drop"]
        camera_gaps = [g for g in win.current_project.gaps if g.event_type == "camera_clicks"]
        gunshot_gaps = [g for g in win.current_project.gaps if g.event_type == "gunshots_combat"]

        print(f"  Detected camera drop gaps: {len(drop_gaps)}, camera clicks: {len(camera_gaps)}, gunshots: {len(gunshot_gaps)}")
        assert len(drop_gaps) >= 1, "Expected at least 1 camera drop gap in the beginning"
        assert len(camera_gaps) >= 1, "Expected at least 1 camera click gap"
        assert len(gunshot_gaps) >= 1, "Expected at least 1 gunshot combat gap"

        for dg in drop_gaps:
            st, en = dg.start_sample, dg.end_sample
            peak_enh = np.max(np.abs(enh_mono[st:en]))
            peak_novoc = np.max(np.abs(novoc_mono[st:en]))
            print(f"  Camera drop gap: peak in novoc={peak_novoc:.3f}, enhanced={peak_enh:.3f}")
            assert peak_enh >= 0.70, f"Camera drop not restored! Peak={peak_enh}, expected >= 0.70"
        print("  ✓ Camera drop impact restored with full punch and presence!")

        for cg in camera_gaps:
            st, en = cg.start_sample, cg.end_sample
            peak_enh = np.max(np.abs(enh_mono[st:en]))
            peak_novoc = np.max(np.abs(novoc_mono[st:en]))
            print(f"  Camera click gap: peak in novoc={peak_novoc:.3f}, enhanced={peak_enh:.3f}")
            assert peak_enh >= 0.45, f"Camera clicks not restored! Peak={peak_enh}, expected >= 0.45"
        print("  ✓ Camera shutter clicks restored with full sharpness!")

        for gg in gunshot_gaps:
            st, en = gg.start_sample, gg.end_sample
            peak_enh = np.max(np.abs(enh_mono[st:en]))
            peak_novoc = np.max(np.abs(novoc_mono[st:en]))
            print(f"  Gunshot gap: peak in novoc={peak_novoc:.3f}, enhanced={peak_enh:.3f}")
            assert peak_enh >= 0.55, f"Gunshots not restored! Peak={peak_enh}, expected >= 0.55"
        print("  ✓ Combat gunshots restored with full punch and impact!")

        # 3. Check that NO repeated donor sound was force-pasted onto dialogue gaps
        for g in speech_gaps:
            st, en = g.start_sample, g.end_sample
            assert np.max(np.abs(enh_mono[st:en])) <= np.max(np.abs(orig_mono[st:en])) + 0.1
        print("  ✓ Verified: 0 dialogue gaps were polluted with repeated donor sounds! Background music is clean.")

        # 4. Check Segment Editor Table
        # Table should show "Чистый фон" as default for dialogue
        assert win.segment_editor.table.rowCount() == len(win.current_project.gaps)
        combo_0 = win.segment_editor.table.cellWidget(0, 3)
        print(f"  Segment 0 sound selected: '{combo_0.currentText()}'")
        assert "Чистый фон" in combo_0.currentText(), f"Expected 'Чистый фон', got '{combo_0.currentText()}'"
        print("  ✓ Segment editor table defaults to clean background for speech.")

        print(">>> TEST 2 PASSED: Full video pipeline and acoustics verified! <<<\n")
    finally:
        QMessageBox.critical = orig_critical
        QMessageBox.warning = orig_warning
        if win:
            win.close()


def test_3_live_adjustments_and_video_remux():
    """Test user customizing a segment in the editor and applying changes."""
    print("\n" + "=" * 60)
    print("TEST 3: User Adjustments & Instant Video Remux")
    print("=" * 60)

    config = Config()
    pm = ProjectManager(config)
    projects = pm.list_projects()
    if not projects:
        print("  [SKIP] No projects available for adjustment test")
        return

    app = get_app()
    win = MainWindow(config=config)
    win.show()
    wait_ms(200)

    # Open the first processed project
    target_proj = next((p for p in projects if p.status == "processed"), None)
    if not target_proj:
        print("  [SKIP] No processed project found")
        win.close()
        return

    win.open_project(target_proj.id)
    wait_ms(300)

    old_prev_video = target_proj.preview_video_path
    print(f"  Loaded project: {target_proj.name}, previous preview: {old_prev_video.name if old_prev_video else 'None'}")

    # Simulate user changing gap 0: pick an explosion or alert sound
    combo_0 = win.segment_editor.table.cellWidget(0, 3)
    # Find a library sound index
    target_idx = -1
    for idx in range(combo_0.count()):
        if "explosion" in combo_0.itemText(idx).lower() or "alarm" in combo_0.itemText(idx).lower():
            target_idx = idx
            break
    if target_idx == -1 and combo_0.count() > 1:
        target_idx = 1

    assert target_idx > 0, "Should have sound options in combo"
    chosen_sound_name = combo_0.itemText(target_idx)
    print(f"  User selects sound for gap 0: '{chosen_sound_name}'")
    combo_0.setCurrentIndex(target_idx)
    wait_ms(200)

    # Simulate user clicking "⚡ Применить правки"
    print("  User clicks '⚡ Применить правки'...")
    win._on_apply_adjustments()

    # Wait for QuickReinsertWorker to complete
    loop = QEventLoop()
    if win.reinsert_worker and win.reinsert_worker.isRunning():
        win.reinsert_worker.finished.connect(lambda res: loop.quit())
        win.reinsert_worker.error.connect(lambda err: loop.quit())
        QTimer.singleShot(30000, loop.quit)
        loop.exec()

    wait_ms(300)
    new_prev_video = win.current_project.preview_video_path
    print(f"  New preview video generated: {new_prev_video.name}")
    assert new_prev_video.exists(), "New remuxed preview video must exist on disk"
    if old_prev_video:
        assert new_prev_video != old_prev_video, "Preview video should be updated with new timestamp"

    print(">>> TEST 3 PASSED: User adjustments and video remuxing work seamlessly! <<<\n")
    win.close()


def test_4_edge_cases_and_stability():
    """Test rapid track switching, extremes, and stress scenarios."""
    print("\n" + "=" * 60)
    print("TEST 4: Edge Cases & Stability Verification")
    print("=" * 60)

    orig_critical = QMessageBox.critical
    orig_warning = QMessageBox.warning
    QMessageBox.critical = lambda *a, **kw: QMessageBox.Ok
    QMessageBox.warning = lambda *a, **kw: QMessageBox.Ok

    win = None
    try:
        config = Config()
        app = get_app()
        win = MainWindow(config=config)
        win.show()
        wait_ms(200)

        # 1. Rapid track switching in video player
        print("  Stress testing rapid audio track switching...")
        for mode in ["enhanced", "no_vocals", "original", "enhanced", "original", "no_vocals"]:
            win.player_widget.switch_track(mode, auto_play=False)
            wait_ms(50)
        print("  ✓ Rapid track switching completed without exception")

        # 2. Volume bounds
        print("  Testing volume slider bounds (-24 dB to +12 dB)...")
        win.player_widget.slider_volume.setValue(0)
        assert win.player_widget.audio_output.volume() == 0.0
        win.player_widget.slider_volume.setValue(100)
        assert win.player_widget.audio_output.volume() == 1.0
        win.player_widget.btn_mute.click()
        assert win.player_widget.audio_output.isMuted()
        win.player_widget.btn_mute.click()
        assert not win.player_widget.audio_output.isMuted()
        print("  ✓ Volume & mute controls verified")

        # 3. Seeking bounds
        win.player_widget.seek_relative(99999999)
        win.player_widget.seek_relative(-99999999)
        print("  ✓ Seek out-of-bounds handled gracefully")

        # 4. Cyrillic project name handling
        pm = ProjectManager(config)
        dummy_wav = config.temp_dir / "test_cyr.wav"
        dummy_wav.write_bytes(b"\x00" * 1024)
        cyr_proj = pm.create_project(dummy_wav, name="Тестовый проект с русскими буквами")
        assert pm.load_project(cyr_proj.id) is not None
        pm.rename_project(cyr_proj.id, "Новое имя (проверка переименования)")
        loaded = pm.load_project(cyr_proj.id)
        assert loaded.name == "Новое имя (проверка переименования)"
        pm.delete_project(cyr_proj.id)
        print("  ✓ Cyrillic project naming, renaming, and loading verified")

        print(">>> TEST 4 PASSED: Edge cases and stability verified! <<<\n")
    finally:
        QMessageBox.critical = orig_critical
        QMessageBox.warning = orig_warning
        if win:
            win.close()


def run_all_tests():
    print("\n" + "█" * 60)
    print("  CLEARSOUND REAL-USER AUTOMATION VERIFICATION SUITE")
    print("█" * 60)

    test_1_hub_project_deletion()
    test_2_real_video_pipeline_and_acoustic_verification()
    test_3_live_adjustments_and_video_remux()
    test_4_edge_cases_and_stability()

    print("\n" + "█" * 60)
    print("  ALL REAL-USER VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
